import argparse
import inspect
import sys
import types
import re
from collections import OrderedDict

import torch


# =========================
# TODO: 너 코드베이스에 맞게 수정
# =========================
# 예시 1) 같은 폴더에 elsnet.py가 있으면:
# from elsnet import ELSNet
#
# 예시 2) mmseg 내부 백본으로 넣어둔 경우(가정):
# from mmseg.models.backbones.elsnet import ELSNet
#
# 예시 3) 네 커스텀 패키지 경로:
# from my_pkg.models.backbones.elsnet import ELSNet
from mmseg.models.backbones.elsnet import ELSNet


def _try_load_with_mmengine(path: str, map_location: str = "cpu"):
    try:
        from mmengine.runner import CheckpointLoader  # noqa
    except Exception as e:
        return None, e

    try:
        ckpt = CheckpointLoader.load_checkpoint(path, map_location=map_location)
        return ckpt, None
    except Exception as e:
        return None, e


def _install_safe_globals_for_weights_only():
    """
    mmengine 체크포인트를 torch.load(weights_only=True)로 열 때 자주 필요한 allowlist.
    (torch/numpy 버전에 따라 추가가 필요할 수도 있음)
    """
    from torch.serialization import add_safe_globals

    # 1) mmengine HistoryBuffer (mmengine이 없으면 더미로 대체)
    module_name = "mmengine.logging.history_buffer"
    try:
        from mmengine.logging.history_buffer import HistoryBuffer  # type: ignore
        add_safe_globals([HistoryBuffer])
    except Exception:
        # 더미 모듈/클래스 생성 후 allowlist
        parts = module_name.split(".")
        acc = ""
        for p in parts:
            acc = p if not acc else acc + "." + p
            if acc not in sys.modules:
                sys.modules[acc] = types.ModuleType(acc)

        class HistoryBuffer:  # dummy
            pass

        HistoryBuffer.__module__ = module_name
        setattr(sys.modules[module_name], "HistoryBuffer", HistoryBuffer)
        add_safe_globals([HistoryBuffer])

    # 2) numpy 관련 (mmengine 메시지허브/스케줄러 쪽에 같이 들어가는 경우 많음)
    import numpy as np
    from numpy.core.multiarray import _reconstruct, scalar  # type: ignore

    add_safe_globals([_reconstruct, scalar, np.ndarray, np.dtype])

    # numpy 2.x 계열에서 dtype가 numpy.dtype[float64]처럼 파라메트릭 타입으로 나오는 경우가 있어 추가 allowlist
    dtype_classes = {type(np.dtype(t)) for t in [
        "float64", "float32", "float16",
        "int64", "int32", "int16", "int8",
        "uint64", "uint32", "uint16", "uint8",
        "bool"
    ]}
    add_safe_globals(list(dtype_classes))


def _try_load_with_torch_weights_only(path: str, map_location: str = "cpu"):
    # torch.load에 weights_only가 있는 버전만 가능
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError(
            "Your torch.load() does not support weights_only=True. "
            "Install mmengine and use CheckpointLoader instead."
        )

    _install_safe_globals_for_weights_only()

    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    return ckpt


def load_checkpoint_any(path: str, map_location: str = "cpu"):
    """
    1) mmengine로 로드 시도
    2) 실패하면 torch weights_only 로드 시도
    """
    ckpt, err = _try_load_with_mmengine(path, map_location=map_location)
    if ckpt is not None:
        return ckpt

    # mmengine 없는 환경에서도 '가중치 변환'만 하고 싶을 때의 fallback
    try:
        return _try_load_with_torch_weights_only(path, map_location=map_location)
    except Exception as e:
        raise RuntimeError(
            "Failed to load checkpoint.\n"
            f"- mmengine load error: {repr(err)}\n"
            f"- torch weights_only load error: {repr(e)}\n\n"
            "Recommendation:\n"
            "  (1) Install mmengine, then rerun.\n"
            "  (2) Or run this script in the same env where you run mmseg training.\n"
            "Security note: Avoid torch.load(weights_only=False) on untrusted .pth."
        )


def extract_state_dict(ckpt):
    """
    mmengine/mmcv 스타일 체크포인트든, raw state_dict든 모두 대응.
    """
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # 이미 state_dict일 가능성
        # (value가 tensor인 dict)
        if all(torch.is_tensor(v) for v in ckpt.values() if v is not None):
            return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def strip_prefixes(key: str, prefixes):
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def convert(
    src_path: str,
    dst_path: str,
    in_channels: int,
    channels: int,
    ppm_channels: int,
    num_stem_blocks: int,
    num_branch_blocks: int,
    drop_prefixes,
    drop_key_regexes,
    save_format: str,
    map_location: str = "cpu",
):
    # 1) target model (shape reference)
    target = ELSNet(
        in_channels=in_channels,
        channels=channels,
        ppm_channels=ppm_channels,
        num_stem_blocks=num_stem_blocks,
        num_branch_blocks=num_branch_blocks,
        init_cfg=None,  # 중요: 여기서 또 pretrained 로드하지 않게
    )
    target_sd = target.state_dict()

    # 2) load source checkpoint
    ckpt = load_checkpoint_any(src_path, map_location=map_location)
    src_sd = extract_state_dict(ckpt)

    # 3) convert
    out_sd = OrderedDict()
    stats = {
        "total_src": 0,
        "dropped_by_regex": 0,
        "missing_in_target": 0,
        "shape_mismatch": 0,
        "kept": 0,
    }
    shape_mismatch_examples = []

    compiled_drop = [re.compile(p) for p in drop_key_regexes]

    for k, v in src_sd.items():
        stats["total_src"] += 1

        nk = strip_prefixes(k, drop_prefixes)

        # drop patterns (decode_head 등)
        if any(r.match(nk) for r in compiled_drop):
            stats["dropped_by_regex"] += 1
            continue

        if nk not in target_sd:
            stats["missing_in_target"] += 1
            continue

        if tuple(v.shape) != tuple(target_sd[nk].shape):
            stats["shape_mismatch"] += 1
            if len(shape_mismatch_examples) < 30:
                shape_mismatch_examples.append((k, nk, tuple(v.shape), tuple(target_sd[nk].shape)))
            continue

        out_sd[nk] = v
        stats["kept"] += 1

    # 4) save
    if save_format == "raw":
        torch.save(out_sd, dst_path)
    elif save_format == "mmengine":
        torch.save({"state_dict": out_sd, "meta": {"converted_from": src_path}}, dst_path)
    else:
        raise ValueError(f"Unknown save_format: {save_format}")

    # 5) report
    print("=== Conversion done ===")
    print("src:", src_path)
    print("dst:", dst_path)
    print("save_format:", save_format)
    for k in ["total_src", "dropped_by_regex", "missing_in_target", "shape_mismatch", "kept"]:
        print(f"{k:18s}: {stats[k]}")
    if shape_mismatch_examples:
        print("\n[shape mismatch examples] (showing up to 30)")
        for a, b, s1, s2 in shape_mismatch_examples:
            print(" -", a, "->", b, s1, "vs", s2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="input .pth (mmseg/runner checkpoint)")
    ap.add_argument("--dst", required=True, help="output .pth (ELSNet pretrained)")
    ap.add_argument("--in-channels", type=int, default=1)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--ppm-channels", type=int, default=96)
    ap.add_argument("--num-stem-blocks", type=int, default=2)
    ap.add_argument("--num-branch-blocks", type=int, default=3)

    ap.add_argument(
        "--drop-prefixes",
        nargs="*",
        default=["module.", "model.", "backbone."],
        help="prefixes to strip repeatedly (default: module., model., backbone.)",
    )
    ap.add_argument(
        "--drop-key-regexes",
        nargs="*",
        default=[r"^decode_head\.", r"^auxiliary_head\."],
        help="regexes for keys to drop (default drops decode_head/auxiliary_head)",
    )
    ap.add_argument(
        "--save-format",
        choices=["raw", "mmengine"],
        default="raw",
        help="raw: save state_dict only (recommended for your PIDNet-style init_weights). "
             "mmengine: save dict with state_dict/meta.",
    )
    args = ap.parse_args()

    convert(
        src_path=args.src,
        dst_path=args.dst,
        in_channels=args.in_channels,
        channels=args.channels,
        ppm_channels=args.ppm_channels,
        num_stem_blocks=args.num_stem_blocks,
        num_branch_blocks=args.num_branch_blocks,
        drop_prefixes=args.drop_prefixes,
        drop_key_regexes=args.drop_key_regexes,
        save_format=args.save_format,
    )


if __name__ == "__main__":
    main()

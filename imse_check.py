#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_RE = re.compile(r"Iter\(train\)\s*\[\s*(\d+)\s*/\s*(\d+)\]")
KEYVAL_RE = re.compile(r"([A-Za-z][A-Za-z0-9_./\-]*)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def parse_train(log_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    rows = []
    with open(log_path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            max_iters = int(m.group(2))

            kv = {}
            for km in KEYVAL_RE.finditer(line):
                k = km.group(1)
                v = float(km.group(2))
                kv[k] = v

            # loss_imse 없으면 스킵 (혹시 일부 라인에 누락될 수 있어 방어)
            if "loss_imse" not in kv or "loss" not in kv:
                continue

            rows.append({
                "step": step,
                "max_iters": max_iters,
                "loss": kv.get("loss"),
                "loss_imse": kv.get("loss_imse"),
                "lr": kv.get("lr", np.nan),
            })

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Iter(train) 라인에서 loss_imse/loss를 파싱하지 못했습니다.")
    df["imse_ratio"] = df["loss_imse"] / df["loss"]
    return df

def rolling(s: pd.Series, win: int, kind: str = "mean") -> pd.Series:
    if kind == "median":
        return s.rolling(win, min_periods=max(3, win // 5)).median()
    return s.rolling(win, min_periods=max(3, win // 5)).mean()

def segment_stats(df: pd.DataFrame) -> dict:
    # step 기준 20%/60%/20% 구간으로 나눔
    n = len(df)
    a = int(n * 0.2)
    b = int(n * 0.8)
    head = df.iloc[:a] if a >= 5 else df.iloc[:max(5, n//5)]
    tail = df.iloc[b:] if (n - b) >= 5 else df.iloc[-max(5, n//5):]
    mid  = df.iloc[a:b] if (b - a) >= 5 else df.iloc[n//5: -n//5]

    def sdesc(x):
        return {
            "mean": float(np.nanmean(x)),
            "median": float(np.nanmedian(x)),
            "std": float(np.nanstd(x)),
            "p10": float(np.nanpercentile(x, 10)),
            "p90": float(np.nanpercentile(x, 90)),
        }

    return {
        "head": sdesc(head["loss_imse"].values),
        "mid":  sdesc(mid["loss_imse"].values) if len(mid) else None,
        "tail": sdesc(tail["loss_imse"].values),
        "head_ratio": sdesc(head["imse_ratio"].values),
        "tail_ratio": sdesc(tail["imse_ratio"].values),
        "head_step_range": (int(head["step"].min()), int(head["step"].max())),
        "tail_step_range": (int(tail["step"].min()), int(tail["step"].max())),
    }

def slope_on_window(df: pd.DataFrame, col: str, frac: float) -> float:
    # 마지막 frac 구간에서의 선형 기울기(단위: loss_imse per step)
    n = len(df)
    k = max(10, int(n * frac))
    sub = df.iloc[-k:]
    x = sub["step"].values.astype(float)
    y = sub[col].values.astype(float)
    # 결측 제거
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return float("nan")
    # 선형회귀
    coef = np.polyfit(x, y, 1)[0]
    return float(coef)

def find_plateau_step(df: pd.DataFrame, col: str, win: int, eps_rel: float = 0.01) -> int:
    """
    smoothed 값이 이후로 거의 개선되지 않는 시점(대략)을 찾음.
    eps_rel: 최근구간에서 상대 개선이 eps_rel 이하이면 plateau로 봄.
    """
    sm = rolling(df[col], win, kind="median").dropna()
    if len(sm) < max(20, win):
        return int(df["step"].iloc[0])

    # 마지막 20% 구간 상대 개선 계산
    n = len(sm)
    last = sm.iloc[int(n * 0.8):]
    if len(last) < 10:
        return int(df["step"].iloc[0])

    rel_improv = (last.iloc[0] - last.iloc[-1]) / (abs(last.iloc[0]) + 1e-12)
    if rel_improv > eps_rel:
        # 아직 plateau라고 보기 애매
        return -1

    # plateau로 판단되면, "최저치에 근접(최저치+5%)"한 최초 step 찾기
    minv = float(sm.min())
    thr = minv * 1.05
    idx = sm[sm <= thr].index.min()
    return int(df.loc[idx, "step"]) if pd.notna(idx) else -1

def plot(df: pd.DataFrame, outdir: str, win: int):
    os.makedirs(outdir, exist_ok=True)

    # loss_imse
    plt.figure()
    plt.plot(df["step"], df["loss_imse"], alpha=0.25)
    plt.plot(df["step"], rolling(df["loss_imse"], win, kind="median"), linewidth=2.0)
    plt.xlabel("step")
    plt.ylabel("loss_imse")
    plt.title("Train loss_imse (raw + rolling median)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_imse.png"), dpi=160)
    plt.close()

    # ratio
    plt.figure()
    plt.plot(df["step"], df["imse_ratio"], alpha=0.25)
    plt.plot(df["step"], rolling(df["imse_ratio"], win, kind="median"), linewidth=2.0)
    plt.xlabel("step")
    plt.ylabel("loss_imse / total_loss")
    plt.title("loss_imse ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_imse_ratio.png"), dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--outdir", default="out_imse")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--win", type=int, default=50, help="rolling window in log-points (logger interval 기준)")
    args = ap.parse_args()

    df = parse_train(args.log, encoding=args.encoding)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "train_imse.csv"), index=False)

    stats = segment_stats(df)
    last_slope = slope_on_window(df, "loss_imse", frac=0.2)
    last_ratio_slope = slope_on_window(df, "imse_ratio", frac=0.2)
    plateau_step = find_plateau_step(df, "loss_imse", win=args.win, eps_rel=0.01)

    report = []
    report.append("# loss_imse audit")
    report.append(f"- rows(parsed train points): {len(df)}")
    report.append(f"- step range: {int(df.step.min())} ~ {int(df.step.max())}")
    report.append("")
    report.append("## Distribution (loss_imse)")
    report.append(f"- head steps {stats['head_step_range']}: median={stats['head']['median']:.6g}, mean={stats['head']['mean']:.6g}, p10~p90={stats['head']['p10']:.6g}~{stats['head']['p90']:.6g}")
    report.append(f"- tail steps {stats['tail_step_range']}: median={stats['tail']['median']:.6g}, mean={stats['tail']['mean']:.6g}, p10~p90={stats['tail']['p10']:.6g}~{stats['tail']['p90']:.6g}")
    rel = (stats["tail"]["median"] - stats["head"]["median"]) / (abs(stats["head"]["median"]) + 1e-12)
    report.append(f"- median relative change (tail vs head): {rel:.2%}")
    report.append("")
    report.append("## Ratio (loss_imse / total_loss)")
    report.append(f"- head ratio median={stats['head_ratio']['median']:.6g}, tail ratio median={stats['tail_ratio']['median']:.6g}")
    report.append("")
    report.append("## Trend diagnostics")
    report.append(f"- last 20% slope(loss_imse) ≈ {last_slope:.3e} per step (음수면 계속 감소, 양수면 증가/드리프트)")
    report.append(f"- last 20% slope(ratio)     ≈ {last_ratio_slope:.3e} per step")
    if plateau_step == -1:
        report.append("- plateau: 아직 '완전 정체'로 보기는 애매(최근 20%에서 감소가 존재)")
    elif plateau_step > 0:
        report.append(f"- plateau: loss_imse가 최저치 근처(≤ min*1.05)에 처음 도달한 step ≈ {plateau_step}")
    else:
        report.append("- plateau: 데이터가 부족하거나 판정 불가")

    # 간단 판정 문구
    report.append("")
    report.append("## Practical verdict")
    if rel < -0.3:
        report.append("- loss_imse는 초기 대비 충분히 감소했습니다 → SDM 관련 목적이 '최적화'된 정황이 강합니다.")
    else:
        report.append("- loss_imse 감소가 뚜렷하지 않습니다 → (1) 이미 매우 빠르게 수렴했거나 (2) weight/gradient가 약하거나 (3) 로스 정의상 감소가 목표가 아닐 수 있습니다.")

    with open(os.path.join(args.outdir, "report_imse.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    plot(df, os.path.join(args.outdir, "plots"), win=args.win)

    print("[OK] wrote:", os.path.join(args.outdir, "train_imse.csv"))
    print("[OK] wrote:", os.path.join(args.outdir, "report_imse.md"))
    print("[OK] plots:", os.path.join(args.outdir, "plots"))

if __name__ == "__main__":
    main()

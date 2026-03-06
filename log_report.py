#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# utils: parsing / flattening
# --------------------------
def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep=sep))
        else:
            out[key] = v
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(flatten_dict(obj))
    return records


def normalize_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    # Case A) tag/step/value 형태 (일부 scalar 로그)
    sample = records[0]
    if "tag" in sample and ("step" in sample or "iter" in sample) and ("value" in sample or "data" in sample):
        df = pd.DataFrame(records)
        step_col = "step" if "step" in df.columns else "iter"
        val_col = "value" if "value" in df.columns else "data"
        df = df[[step_col, "tag", val_col]].rename(columns={step_col: "iter", "tag": "metric", val_col: "value"})
        wide = df.pivot_table(index="iter", columns="metric", values="value", aggfunc="last").reset_index()
        return wide

    # Case B) mmengine/mmcv *.log.json 형태
    df = pd.DataFrame(records)
    if "iter" not in df.columns:
        for c in ["iteration", "step", "global_step"]:
            if c in df.columns:
                df = df.rename(columns={c: "iter"})
                break
    return df


def try_parse_eta_to_seconds(df: pd.DataFrame) -> None:
    if "eta" in df.columns and df["eta"].dtype == object:
        td = pd.to_timedelta(df["eta"], errors="coerce")
        if td.notna().any():
            df["eta_sec"] = td.dt.total_seconds()


def lr_to_scalar(v: Any) -> float:
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if isinstance(v, (list, tuple)) and v:
        if isinstance(v[0], (int, float, np.number)):
            return float(v[0])
    if isinstance(v, dict):
        for _, val in v.items():
            x = lr_to_scalar(val)
            if not np.isnan(x):
                return x
    return float("nan")


def ensure_lr_scalar(df: pd.DataFrame) -> None:
    if "lr" in df.columns and not pd.api.types.is_numeric_dtype(df["lr"]):
        df["lr_scalar"] = df["lr"].apply(lr_to_scalar)


def coerce_numeric_columns(df: pd.DataFrame, min_non_nan_ratio: float = 0.1) -> None:
    protected = {"mode", "iter", "epoch"}
    for c in list(df.columns):
        if c in protected:
            continue
        if df[c].dtype == object:
            s = pd.to_numeric(df[c], errors="coerce")
            ratio = float(s.notna().mean())
            if ratio >= min_non_nan_ratio and s.notna().any():
                df[c] = s


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return (
        pd.Series(y)
        .rolling(window, min_periods=max(1, window // 2), center=True)
        .mean()
        .to_numpy()
    )


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    m = (b != 0) & np.isfinite(b) & np.isfinite(a)
    out[m] = a[m] / b[m]
    return out


def pick_numeric_metrics(df: pd.DataFrame) -> List[str]:
    metrics = []
    for c in df.columns:
        if c in {"iter", "epoch", "mode"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            metrics.append(c)
    return metrics


def direction_hint(metric: str) -> str:
    m = metric.lower()
    if "loss" in m or "error" in m:
        return "min"
    if "miou" in m or re.search(r"\biou\b", m) or "acc" in m or "dice" in m or "fscore" in m:
        return "max"
    return "none"


# --------------------------
# turning points / plateaus
# --------------------------
def detect_turning_points(x: np.ndarray, y: np.ndarray, min_abs_change: float = 0.0) -> List[int]:
    dy = np.diff(y)
    sign = np.sign(dy)
    sign[np.abs(dy) < min_abs_change] = 0

    idx: List[int] = []
    prev: Optional[float] = None
    for i, s in enumerate(sign):
        if s == 0:
            continue
        if prev is None:
            prev = s
            continue
        if s != prev:
            idx.append(i + 1)
            prev = s
    return idx


def detect_plateaus(
    x: np.ndarray, y: np.ndarray, window: int = 50, slope_eps: float = 1e-5, min_len: int = 100
) -> List[Dict[str, Any]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < window:
        return []

    slopes = np.full(n, np.nan)
    for i in range(window - 1, n):
        xs = x[i - window + 1 : i + 1]
        ys = y[i - window + 1 : i + 1]
        if np.any(np.isnan(ys)):
            continue
        vx = xs - xs.mean()
        denom = float(np.dot(vx, vx))
        if denom == 0:
            continue
        slopes[i] = float(np.dot(vx, ys - ys.mean()) / denom)

    flat = np.abs(slopes) < slope_eps
    segs = []
    start = None
    for i, flag in enumerate(flat):
        if flag and start is None:
            start = i
        if (not flag or i == n - 1) and start is not None:
            end = i if not flag else i
            if end - start + 1 >= min_len:
                segs.append((start, end))
            start = None

    out = []
    for s, e in segs:
        out.append(
            {
                "start_iter": int(x[s]),
                "end_iter": int(x[e]),
                "len": int(e - s + 1),
                "slope": float(np.nanmean(slopes[s : e + 1])),
            }
        )
    return out


def compute_metric_summary(
    df: pd.DataFrame,
    metric: str,
    smooth_window: int,
    turn_min_change: float,
    plateau_window: int,
    plateau_slope_eps: float,
    plateau_min_len: int,
) -> Optional[Dict[str, Any]]:
    d = df[["iter", metric]].dropna()
    if d.empty:
        return None

    x = d["iter"].to_numpy()
    y = d[metric].to_numpy(dtype=float)
    ys = smooth(y, smooth_window)

    hint = direction_hint(metric)
    if hint == "min":
        best_idx = int(np.nanargmin(ys))
    elif hint == "max":
        best_idx = int(np.nanargmax(ys))
    else:
        best_idx = len(x) - 1

    turning = detect_turning_points(x, ys, min_abs_change=turn_min_change)
    plateaus = detect_plateaus(
        x, ys, window=plateau_window, slope_eps=plateau_slope_eps, min_len=plateau_min_len
    )

    k = min(50, len(x))
    slope_last50 = float("nan")
    if k >= 2:
        xs = x[-k:].astype(float)
        ys2 = ys[-k:]
        vx = xs - xs.mean()
        denom = float(np.dot(vx, vx))
        if denom > 0:
            slope_last50 = float(np.dot(vx, ys2 - ys2.mean()) / denom)

    return {
        "metric": metric,
        "n_points": int(len(x)),
        "first_iter": int(x[0]),
        "last_iter": int(x[-1]),
        "first_value": float(y[0]),
        "last_value": float(y[-1]),
        "best_iter": int(x[best_idx]),
        "best_value": float(ys[best_idx]),
        "delta_last_first": float(y[-1] - y[0]),
        "slope_last50": slope_last50,
        "turning_points": int(len(turning)),
        "turning_iters": ",".join(str(int(x[i])) for i in turning[:10]) + ("..." if len(turning) > 10 else ""),
        "plateau_segments": int(len(plateaus)),
        "plateau_iters": "; ".join(
            f"{p['start_iter']}-{p['end_iter']} (len={p['len']})" for p in plateaus[:5]
        )
        + ("..." if len(plateaus) > 5 else ""),
    }


def df_to_markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "(empty)"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = []
    lines.append("|" + "|".join(cols) + "|")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in view.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("|" + "|".join(vals) + "|")
    if len(df) > max_rows:
        lines.append(f"\n(Showing first {max_rows} rows of {len(df)}.)")
    return "\n".join(lines)


# --------------------------
# plotting helpers
# --------------------------
def plot_metrics_grid(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    smooth_window: int,
    ncols: int,
    dpi: int,
) -> None:
    if not metrics:
        return
    df = df.sort_values("iter")
    n = len(metrics)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(6 * ncols, 3.2 * nrows))
    for i, m in enumerate(metrics, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        d = df[["iter", m]].dropna()
        if d.empty:
            ax.set_axis_off()
            continue
        x = d["iter"].to_numpy()
        y = d[m].to_numpy(dtype=float)

        ax.plot(x, y, linewidth=1, alpha=0.5, label="raw")
        if smooth_window > 1 and len(y) >= smooth_window:
            ys = smooth(y, smooth_window)
            ax.plot(x, ys, linewidth=1.5, label=f"smooth(w={smooth_window})")
            ax.legend(fontsize=8)

        ax.set_title(m)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_performance(
    df_train: pd.DataFrame,
    out_path: Path,
    dpi: int,
    smooth_window: int,
    samples_per_iter: Optional[float],
    clip_max_norm: Optional[float],
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    df = df_train.sort_values("iter").copy()

    needed = []
    for c in ["time", "data_time", "memory"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            needed.append(c)

    # iter/sec, samples/sec
    if "time" in df.columns and pd.api.types.is_numeric_dtype(df["time"]):
        t = df["time"].to_numpy(dtype=float)
        df["iter_per_sec"] = safe_div(np.ones_like(t), t)
        if samples_per_iter is not None:
            df["samples_per_sec"] = safe_div(np.full_like(t, samples_per_iter, dtype=float), t)
        else:
            notes.append("samples/sec skipped (provide --samples_per_iter or --samples_per_gpu+--world_size).")
    else:
        notes.append("iter/sec skipped (no numeric 'time' column).")

    # time breakdown ratio
    if "time" in df.columns and "data_time" in df.columns and pd.api.types.is_numeric_dtype(df["time"]) and pd.api.types.is_numeric_dtype(df["data_time"]):
        df["data_ratio"] = safe_div(df["data_time"].to_numpy(dtype=float), df["time"].to_numpy(dtype=float))
        df["gpu_time_est"] = df["time"] - df["data_time"]
    else:
        notes.append("time breakdown skipped (need numeric 'time' and 'data_time').")

    # grad_norm + clip ratio
    grad_cols = [c for c in df.columns if re.search(r"grad.*norm", c, re.IGNORECASE)]
    if grad_cols:
        # pick the most "generic" one first
        grad_cols = sorted(grad_cols, key=lambda x: (0 if x.lower() in {"grad_norm", "gradnorm"} else 1, len(x)))
        gc = grad_cols[0]
        df["_grad_norm_used"] = df[gc]
        if clip_max_norm is not None:
            df["clip_ratio"] = safe_div(df["_grad_norm_used"].to_numpy(dtype=float), np.full(len(df), clip_max_norm, dtype=float))
        else:
            notes.append("clip ratio skipped (provide --clip_max_norm).")
        notes.append(f"grad_norm source column: {gc}")
    else:
        notes.append("grad_norm skipped (no column matching /grad.*norm/i).")

    # plotting: 2x2 or 3x2 depending on availability
    candidates = []
    if "iter_per_sec" in df.columns:
        candidates.append("iter_per_sec")
    if "samples_per_sec" in df.columns:
        candidates.append("samples_per_sec")
    if "data_ratio" in df.columns:
        candidates.append("data_ratio")
    if "time" in df.columns and "data_time" in df.columns:
        candidates.append("time_vs_data_time")
    if "memory" in df.columns:
        candidates.append("memory")
    if "_grad_norm_used" in df.columns:
        candidates.append("grad_norm")
    if "clip_ratio" in df.columns:
        candidates.append("clip_ratio")

    n = len(candidates)
    if n == 0:
        return pd.DataFrame(), notes

    ncols = 2
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(12, 3.2 * nrows))

    x = df["iter"].to_numpy(dtype=float)
    for i, item in enumerate(candidates, start=1):
        ax = fig.add_subplot(nrows, ncols, i)

        if item == "time_vs_data_time":
            y1 = df["time"].to_numpy(dtype=float)
            y2 = df["data_time"].to_numpy(dtype=float)
            ax.plot(x, y1, linewidth=1, alpha=0.8, label="time")
            ax.plot(x, y2, linewidth=1, alpha=0.8, label="data_time")
            if smooth_window > 1 and len(y1) >= smooth_window:
                ax.plot(x, smooth(y1, smooth_window), linewidth=1.5, label=f"time smooth(w={smooth_window})")
                ax.plot(x, smooth(y2, smooth_window), linewidth=1.5, label=f"data_time smooth(w={smooth_window})")
            ax.set_title("time & data_time")
            ax.legend(fontsize=8)
        elif item == "memory":
            y = df["memory"].to_numpy(dtype=float)
            ax.plot(x, y, linewidth=1)
            ax.set_title("memory (as logged)")
        elif item == "grad_norm":
            y = df["_grad_norm_used"].to_numpy(dtype=float)
            ax.plot(x, y, linewidth=1, alpha=0.8, label="grad_norm")
            if smooth_window > 1 and len(y) >= smooth_window:
                ax.plot(x, smooth(y, smooth_window), linewidth=1.5, label=f"smooth(w={smooth_window})")
                ax.legend(fontsize=8)
            ax.set_title("grad_norm")
        else:
            y = df[item].to_numpy(dtype=float)
            ax.plot(x, y, linewidth=1, alpha=0.8, label=item)
            if smooth_window > 1 and len(y) >= smooth_window:
                ax.plot(x, smooth(y, smooth_window), linewidth=1.5, label=f"smooth(w={smooth_window})")
                ax.legend(fontsize=8)
            ax.set_title(item)

        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # performance summary stats
    stats_rows = []
    for c in ["iter_per_sec", "samples_per_sec", "time", "data_time", "data_ratio", "memory", "_grad_norm_used", "clip_ratio"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            v = df[c].dropna().to_numpy(dtype=float)
            if v.size == 0:
                continue
            stats_rows.append({
                "metric": c,
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p90": float(np.quantile(v, 0.90)),
                "p95": float(np.quantile(v, 0.95)),
                "max": float(np.max(v)),
            })
    stats = pd.DataFrame(stats_rows).sort_values("metric") if stats_rows else pd.DataFrame()
    return stats, notes


# --------------------------
# loss breakdown
# --------------------------
def infer_loss_components(df: pd.DataFrame) -> List[str]:
    # numeric columns containing "loss" but not the total "loss" itself
    comps = []
    for c in df.columns:
        if c.lower() == "loss":
            continue
        if "loss" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            comps.append(c)
    return comps


def group_key(name: str) -> str:
    # prefer dot prefix, fallback underscore prefix
    if "." in name:
        return name.split(".", 1)[0]
    if "_" in name:
        return name.split("_", 1)[0]
    return "other"


def plot_loss_breakdown(
    df_train: pd.DataFrame,
    out_path: Path,
    dpi: int,
    smooth_window: int,
    loss_topk: int,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    df = df_train.sort_values("iter").copy()
    x = df["iter"].to_numpy(dtype=float)

    comps = infer_loss_components(df)
    has_total = "loss" in df.columns and pd.api.types.is_numeric_dtype(df["loss"])
    if not comps and not has_total:
        notes.append("loss breakdown skipped (no numeric loss columns).")
        return pd.DataFrame(), notes

    # component matrix
    comp_df = df[["iter"] + comps].copy() if comps else df[["iter"]].copy()
    comp_df = comp_df.dropna(subset=["iter"])

    # group sums
    group_map: Dict[str, List[str]] = {}
    for c in comps:
        g = group_key(c)
        group_map.setdefault(g, []).append(c)

    # compute total for ratios: prefer sum of components, else use 'loss'
    if comps:
        comp_sum = comp_df[comps].sum(axis=1, skipna=True)
        comp_df["_comp_sum"] = comp_sum
        total_for_ratio = comp_sum
        if has_total:
            comp_df["_loss_total"] = df["loss"]
    else:
        total_for_ratio = df["loss"]
        comp_df["_loss_total"] = df["loss"]

    # group ratios
    group_ratios = {}
    for g, cols in group_map.items():
        group_ratios[g] = safe_div(comp_df[cols].sum(axis=1, skipna=True).to_numpy(dtype=float),
                                  total_for_ratio.to_numpy(dtype=float))

    # top-k component ratios by mean contribution
    top_cols: List[str] = []
    if comps:
        means = []
        denom = total_for_ratio.to_numpy(dtype=float)
        for c in comps:
            r = safe_div(comp_df[c].to_numpy(dtype=float), denom)
            means.append((c, float(np.nanmean(r))))
        means.sort(key=lambda t: t[1], reverse=True)
        top_cols = [c for c, _ in means[:max(1, loss_topk)]]
        other_cols = [c for c in comps if c not in top_cols]
        notes.append(f"loss components: {len(comps)} (topk={len(top_cols)}, other={len(other_cols)})")
    else:
        notes.append("only total loss available; component ratios skipped.")

    # plot layout
    panels = 1
    panels += 1 if group_ratios else 0
    panels += 1 if top_cols else 0

    fig = plt.figure(figsize=(12, 3.2 * panels))
    pi = 1

    # (1) total loss
    ax = fig.add_subplot(panels, 1, pi); pi += 1
    if has_total:
        y = df["loss"].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=1, alpha=0.6, label="loss (raw)")
        if smooth_window > 1 and len(y) >= smooth_window:
            ax.plot(x, smooth(y, smooth_window), linewidth=1.6, label=f"loss smooth(w={smooth_window})")
        ax.set_title("total loss")
        ax.legend(fontsize=8)
    else:
        # plot component sum
        y = comp_df["_comp_sum"].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=1, alpha=0.7, label="sum(loss components)")
        if smooth_window > 1 and len(y) >= smooth_window:
            ax.plot(x, smooth(y, smooth_window), linewidth=1.6, label=f"smooth(w={smooth_window})")
        ax.set_title("sum of loss components (no 'loss' column)")
        ax.legend(fontsize=8)
    ax.set_xlabel("iteration"); ax.grid(True, alpha=0.3)

    # (2) group ratios stacked
    if group_ratios:
        ax = fig.add_subplot(panels, 1, pi); pi += 1
        gs = sorted(group_ratios.keys())
        Y = np.vstack([np.nan_to_num(group_ratios[g], nan=0.0) for g in gs])
        ax.stackplot(x, Y, labels=gs, alpha=0.9)
        ax.set_title("loss group ratio (by prefix: decode/aux/...)")
        ax.set_xlabel("iteration")
        ax.legend(fontsize=8, ncol=min(4, len(gs)))
        ax.grid(True, alpha=0.3)

    # (3) top-k component ratios stacked
    if top_cols:
        ax = fig.add_subplot(panels, 1, pi); pi += 1
        denom = total_for_ratio.to_numpy(dtype=float)
        Y_list = []
        labels = []
        for c in top_cols:
            r = safe_div(comp_df[c].to_numpy(dtype=float), denom)
            Y_list.append(np.nan_to_num(r, nan=0.0))
            labels.append(c)
        if other_cols:
            other = np.zeros_like(Y_list[0])
            for c in other_cols:
                r = safe_div(comp_df[c].to_numpy(dtype=float), denom)
                other += np.nan_to_num(r, nan=0.0)
            Y_list.append(other)
            labels.append("other")
        Y = np.vstack(Y_list)
        ax.stackplot(x, Y, labels=labels, alpha=0.9)
        ax.set_title(f"loss component ratio (top {len(top_cols)} + other)")
        ax.set_xlabel("iteration")
        ax.legend(fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # summary table
    rows = []
    if has_total:
        v = df["loss"].dropna().to_numpy(dtype=float)
        if v.size:
            rows.append({"metric": "loss", "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))})
    for g, r in group_ratios.items():
        rr = pd.Series(r).dropna().to_numpy(dtype=float)
        if rr.size:
            rows.append({"metric": f"group_ratio:{g}", "mean": float(np.mean(rr)), "min": float(np.min(rr)), "max": float(np.max(rr))})
    summary = pd.DataFrame(rows).sort_values("metric") if rows else pd.DataFrame()
    return summary, notes


# --------------------------
# per-class IoU/Acc extraction
# --------------------------
def extract_per_class_from_log(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Look for columns like:
      IoU.classname, Acc.classname, iou.xxx, acc.xxx
    Returns dict: {"IoU": df_wide, "Acc": df_wide}
      df_wide index: iter, columns: class names
    """
    out: Dict[str, pd.DataFrame] = {}
    if "iter" not in df.columns:
        return out

    patterns = [
        (re.compile(r"^(IoU|iou)\.(.+)$"), "IoU"),
        (re.compile(r"^(Acc|acc)\.(.+)$"), "Acc"),
        (re.compile(r"^(mIoU|miou)\.(.+)$"), "IoU"),  # fallback
    ]

    cols_iou = {}
    cols_acc = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        for rgx, kind in patterns:
            m = rgx.match(c)
            if m:
                cls = m.group(2)
                if kind == "IoU":
                    cols_iou[cls] = c
                else:
                    cols_acc[cls] = c

    if cols_iou:
        tmp = df[["iter"] + list(cols_iou.values())].copy()
        tmp = tmp.rename(columns={v: k for k, v in cols_iou.items()})
        out["IoU"] = tmp.dropna(subset=["iter"]).sort_values("iter")
    if cols_acc:
        tmp = df[["iter"] + list(cols_acc.values())].copy()
        tmp = tmp.rename(columns={v: k for k, v in cols_acc.items()})
        out["Acc"] = tmp.dropna(subset=["iter"]).sort_values("iter")

    return out


def parse_iter_from_filename(p: Path) -> Optional[int]:
    m = re.findall(r"\d+", p.name)
    if not m:
        return None
    # 가장 큰 숫자를 iter로 가정
    return int(max(m, key=lambda s: int(s)))


def extract_per_class_from_eval_dir(eval_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Best-effort: recursively scan JSON files, try to find per-class IoU/Acc.
    Supports common shapes:
      - {"iter": 8000, "class_names": [...], "IoU": [...], "Acc": [...]}
      - {"class_names": [...], "per_class_iou": [...], "per_class_acc": [...]}
      - {"IoU": {"cls": v, ...}, "Acc": {"cls": v, ...}}
    Returns dict {"IoU": df, "Acc": df} with iter column.
    """
    out_rows_iou = []
    out_rows_acc = []

    json_files = list(eval_dir.rglob("*.json"))
    for p in json_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        it = None
        if "iter" in obj and isinstance(obj["iter"], (int, float)):
            it = int(obj["iter"])
        else:
            it = parse_iter_from_filename(p)

        # dict mapping
        def take_map(key: str) -> Optional[Dict[str, float]]:
            v = obj.get(key, None)
            if isinstance(v, dict):
                d = {}
                for k, vv in v.items():
                    if isinstance(vv, (int, float)) and np.isfinite(vv):
                        d[str(k)] = float(vv)
                return d if d else None
            return None

        iou_map = take_map("IoU") or take_map("iou") or take_map("per_class_iou")
        acc_map = take_map("Acc") or take_map("acc") or take_map("per_class_acc")

        # list+class_names
        class_names = obj.get("class_names") or obj.get("classes") or obj.get("CLASSES")
        if isinstance(class_names, list):
            def take_list(key: str) -> Optional[List[float]]:
                v = obj.get(key, None)
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    return [float(x) for x in v]
                return None

            iou_list = take_list("IoU") or take_list("iou") or take_list("per_class_iou")
            acc_list = take_list("Acc") or take_list("acc") or take_list("per_class_acc")
            if iou_list and len(iou_list) == len(class_names):
                iou_map = {str(c): float(v) for c, v in zip(class_names, iou_list)}
            if acc_list and len(acc_list) == len(class_names):
                acc_map = {str(c): float(v) for c, v in zip(class_names, acc_list)}

        if it is None:
            continue

        if iou_map:
            row = {"iter": it}
            row.update(iou_map)
            out_rows_iou.append(row)
        if acc_map:
            row = {"iter": it}
            row.update(acc_map)
            out_rows_acc.append(row)

    out: Dict[str, pd.DataFrame] = {}
    if out_rows_iou:
        df_iou = pd.DataFrame(out_rows_iou).dropna(subset=["iter"]).sort_values("iter")
        df_iou = df_iou.groupby("iter", as_index=False).last()
        out["IoU"] = df_iou
    if out_rows_acc:
        df_acc = pd.DataFrame(out_rows_acc).dropna(subset=["iter"]).sort_values("iter")
        df_acc = df_acc.groupby("iter", as_index=False).last()
        out["Acc"] = df_acc
    return out


def plot_per_class_heatmap(df_wide: pd.DataFrame, out_path: Path, title: str, dpi: int) -> None:
    """
    df_wide: columns = ['iter', class1, class2, ...]
    """
    if df_wide.empty or df_wide.shape[1] <= 2:
        return
    df_wide = df_wide.sort_values("iter")
    iters = df_wide["iter"].to_numpy(dtype=int)
    classes = [c for c in df_wide.columns if c != "iter"]
    mat = df_wide[classes].to_numpy(dtype=float).T  # (C, T)

    fig = plt.figure(figsize=(12, max(4, 0.28 * len(classes))))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")  # default colormap
    ax.set_title(title)
    ax.set_xlabel("eval step index (sorted by iter)")
    ax.set_ylabel("class")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, fontsize=7)

    # x ticks: show a few iterations
    nt = len(iters)
    ticks = np.linspace(0, max(0, nt - 1), num=min(10, nt), dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(iters[i]) for i in ticks], rotation=45, ha="right", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# checkpoint candidates (recommendation only)
# --------------------------
def list_checkpoints(ckpt_dir: Path) -> pd.DataFrame:
    pths = list(ckpt_dir.rglob("*.pth"))
    rows = []
    for p in pths:
        it = None
        # common patterns: iter_8000.pth, epoch_80.pth
        m = re.search(r"iter_(\d+)", p.name)
        if m:
            it = int(m.group(1))
        else:
            m = re.search(r"epoch_(\d+)", p.name)
            if m:
                it = None  # epoch-based; mapping handled separately (not robust without epoch->iter map)
        rows.append({"checkpoint": str(p), "iter_from_name": it})
    return pd.DataFrame(rows)


def pick_primary_metric(val_df: pd.DataFrame, regex: str) -> Optional[str]:
    if val_df.empty:
        return None
    rgx = re.compile(regex)
    candidates = [c for c in val_df.columns if c not in {"iter", "mode", "epoch"} and rgx.search(c)]
    # prefer mIoU if exists
    for key in ["mIoU", "miou", "m_iou"]:
        for c in candidates:
            if key.lower() in c.lower():
                return c
    return candidates[0] if candidates else None


def map_checkpoints_to_metric(
    ckpt_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metric: str
) -> pd.DataFrame:
    if ckpt_df.empty or val_df.empty or metric not in val_df.columns:
        return pd.DataFrame()

    v = val_df[["iter", metric]].dropna().sort_values("iter")
    vit = v["iter"].to_numpy(dtype=int)
    vval = v[metric].to_numpy(dtype=float)

    rows = []
    for _, r in ckpt_df.iterrows():
        it = r["iter_from_name"]
        if it is None:
            continue
        # find closest val iter <= ckpt iter
        idx = np.searchsorted(vit, it, side="right") - 1
        if idx < 0:
            continue
        rows.append({
            "checkpoint": r["checkpoint"],
            "ckpt_iter": int(it),
            "matched_val_iter": int(vit[idx]),
            metric: float(vval[idx]),
            "gap_iter": int(it - vit[idx]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # rank by metric direction
    hint = direction_hint(metric)
    asc = True if hint == "min" else False
    out = out.sort_values(metric, ascending=asc).reset_index(drop=True)
    return out


# --------------------------
# main
# --------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to *.log.json (JSON lines) or scalar jsonl")
    ap.add_argument("--out", default="log_report_plus", help="output directory")

    ap.add_argument("--include", default="", help="regex include metrics (optional)")
    ap.add_argument("--exclude", default="", help="regex exclude metrics (optional)")
    ap.add_argument("--smooth", type=int, default=21, help="moving average window (train)")
    ap.add_argument("--dpi", type=int, default=340, help="output dpi for PNG")
    ap.add_argument("--ncols", type=int, default=3, help="columns in subplot grid")

    ap.add_argument("--turn_min_change", type=float, default=0.0, help="min abs change to count turning point")
    ap.add_argument("--plateau_window", type=int, default=50)
    ap.add_argument("--plateau_slope_eps", type=float, default=1e-5)
    ap.add_argument("--plateau_min_len", type=int, default=100)

    # throughput
    ap.add_argument("--samples_per_iter", type=float, default=None, help="global samples per iteration (optional)")
    ap.add_argument("--samples_per_gpu", type=float, default=None, help="samples per gpu (optional)")
    ap.add_argument("--world_size", type=int, default=1, help="num gpus (for samples/sec)")

    # grad clip
    ap.add_argument("--clip_max_norm", type=float, default=None, help="max_norm used for grad clipping (optional)")

    # loss breakdown
    ap.add_argument("--loss_topk", type=int, default=6, help="top-k loss components to show")

    # per-class
    ap.add_argument("--eval_dir", default="", help="directory to scan for per-class eval json (optional)")

    # checkpoint recommendation table
    ap.add_argument("--ckpt_dir", default="", help="directory to scan for checkpoints (*.pth)")
    ap.add_argument("--primary_metric", default="mIoU|aAcc|mAcc|IoU|Acc", help="regex to choose primary val metric")
    ap.add_argument("--topk_ckpt", type=int, default=5, help="top K checkpoints to show in report")

    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # samples_per_iter resolution
    samples_per_iter = args.samples_per_iter
    if samples_per_iter is None and args.samples_per_gpu is not None:
        samples_per_iter = float(args.samples_per_gpu) * float(max(1, args.world_size))

    records = read_jsonl(log_path)
    if not records:
        raise SystemExit(f"No JSON records found: {log_path}")

    df = normalize_records(records)
    if df.empty:
        raise SystemExit("Parsed dataframe is empty.")
    if "iter" not in df.columns:
        raise SystemExit("Cannot find iteration column (iter/iteration/step/global_step).")

    df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
    df = df.dropna(subset=["iter"])
    try_parse_eta_to_seconds(df)
    ensure_lr_scalar(df)
    coerce_numeric_columns(df, min_non_nan_ratio=0.1)

    # split train/val
    if "mode" in df.columns:
        mode = df["mode"].astype(str)
        train = df[mode.str.contains("train", case=False, na=False)].copy()
        val = df[mode.str.contains("val|test", case=False, na=False)].copy()
    else:
        train = df.copy()
        val = pd.DataFrame()

    # metric filters
    inc = re.compile(args.include) if args.include else None
    exc = re.compile(args.exclude) if args.exclude else None

    def filter_metrics(metrics: List[str]) -> List[str]:
        out = []
        for m in metrics:
            if inc and not inc.search(m):
                continue
            if exc and exc.search(m):
                continue
            out.append(m)
        return out

    train_metrics = filter_metrics(pick_numeric_metrics(train))
    val_metrics = filter_metrics(pick_numeric_metrics(val)) if not val.empty else []

    # core plots
    if train_metrics:
        plot_metrics_grid(train, train_metrics, out_dir / "train_metrics.png", args.smooth, args.ncols, args.dpi)
    if val_metrics:
        plot_metrics_grid(val, val_metrics, out_dir / "val_metrics.png", 1, args.ncols, args.dpi)

    # metric summaries
    rows = []
    for m in train_metrics:
        s = compute_metric_summary(
            train, m, args.smooth, args.turn_min_change, args.plateau_window, args.plateau_slope_eps, args.plateau_min_len
        )
        if s:
            s["mode"] = "train"
            rows.append(s)
    for m in val_metrics:
        s = compute_metric_summary(
            val, m, 1, args.turn_min_change, max(5, args.plateau_window // 5), args.plateau_slope_eps, max(5, args.plateau_min_len // 5)
        )
        if s:
            s["mode"] = "val"
            rows.append(s)

    metric_summary = pd.DataFrame(rows).sort_values(["mode", "metric"]) if rows else pd.DataFrame()
    metric_summary.to_csv(out_dir / "metric_summary.csv", index=False, encoding="utf-8-sig")

    # performance plot + summary
    perf_summary, perf_notes = plot_performance(
        train, out_dir / "performance.png", args.dpi, args.smooth, samples_per_iter, args.clip_max_norm
    )
    if not perf_summary.empty:
        perf_summary.to_csv(out_dir / "performance_summary.csv", index=False, encoding="utf-8-sig")

    # loss breakdown plot + summary
    loss_summary, loss_notes = plot_loss_breakdown(
        train, out_dir / "loss_breakdown.png", args.dpi, args.smooth, args.loss_topk
    )
    if not loss_summary.empty:
        loss_summary.to_csv(out_dir / "loss_breakdown_summary.csv", index=False, encoding="utf-8-sig")

    # per-class from log first
    per_class = {}
    if not val.empty:
        per_class = extract_per_class_from_log(val)

    # if eval_dir provided, merge/override with eval files (adds more chances)
    eval_notes = []
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
        if eval_dir.exists():
            per2 = extract_per_class_from_eval_dir(eval_dir)
            # merge: prefer eval_dir results if log lacks
            for k, dfk in per2.items():
                if k not in per_class or per_class[k].shape[1] <= 2:
                    per_class[k] = dfk
            if not per2:
                eval_notes.append(f"no per-class structure found under: {eval_dir}")
        else:
            eval_notes.append(f"eval_dir not found: {eval_dir}")

    if "IoU" in per_class and not per_class["IoU"].empty:
        plot_per_class_heatmap(per_class["IoU"], out_dir / "per_class_iou.png", "per-class IoU", args.dpi)
        per_class["IoU"].to_csv(out_dir / "per_class_iou.csv", index=False, encoding="utf-8-sig")
    if "Acc" in per_class and not per_class["Acc"].empty:
        plot_per_class_heatmap(per_class["Acc"], out_dir / "per_class_acc.png", "per-class Acc", args.dpi)
        per_class["Acc"].to_csv(out_dir / "per_class_acc.csv", index=False, encoding="utf-8-sig")

    # checkpoint candidates (recommendation only)
    ckpt_notes = []
    ckpt_candidates = pd.DataFrame()
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        if ckpt_dir.exists():
            ckpts = list_checkpoints(ckpt_dir)
            primary = pick_primary_metric(val, args.primary_metric) if not val.empty else None
            if primary is None:
                ckpt_notes.append("primary metric not found in val logs; checkpoint mapping skipped.")
            else:
                cand = map_checkpoints_to_metric(ckpts, val, primary)
                ckpt_candidates = cand
                if not cand.empty:
                    cand.to_csv(out_dir / "checkpoint_candidates.csv", index=False, encoding="utf-8-sig")
                ckpt_notes.append(f"primary metric for checkpoint table: {primary}")
        else:
            ckpt_notes.append(f"ckpt_dir not found: {ckpt_dir}")

    # report.md
    md = []
    md.append("# MMSeg training log report (plus)\n")
    md.append(f"- Source: `{log_path}`")
    md.append(f"- Output dir: `{out_dir}`")
    if samples_per_iter is not None:
        md.append(f"- samples_per_iter (global): `{samples_per_iter}`")
    md.append("")

    if (out_dir / "train_metrics.png").exists():
        md.append("## Train metrics\n\n![](train_metrics.png)\n")
    if (out_dir / "val_metrics.png").exists():
        md.append("## Val/Test metrics\n\n![](val_metrics.png)\n")

    if (out_dir / "performance.png").exists():
        md.append("## Performance (throughput / time breakdown / grad)\n\n![](performance.png)\n")
        if not perf_summary.empty:
            md.append("### Performance summary\n")
            md.append(df_to_markdown_table(perf_summary, max_rows=50))
            md.append("")
        if perf_notes:
            md.append("### Notes\n")
            for n in perf_notes:
                md.append(f"- {n}")
            md.append("")

    if (out_dir / "loss_breakdown.png").exists():
        md.append("## Loss breakdown\n\n![](loss_breakdown.png)\n")
        if not loss_summary.empty:
            md.append("### Loss breakdown summary\n")
            md.append(df_to_markdown_table(loss_summary, max_rows=80))
            md.append("")
        if loss_notes:
            md.append("### Notes\n")
            for n in loss_notes:
                md.append(f"- {n}")
            md.append("")

    if (out_dir / "per_class_iou.png").exists():
        md.append("## Per-class IoU\n\n![](per_class_iou.png)\n")
    if (out_dir / "per_class_acc.png").exists():
        md.append("## Per-class Acc\n\n![](per_class_acc.png)\n")
    if eval_notes:
        md.append("### Per-class extraction notes\n")
        for n in eval_notes:
            md.append(f"- {n}")
        md.append("")

    md.append("## Metric summary\n")
    md.append(df_to_markdown_table(metric_summary, max_rows=60))
    md.append("")

    # checkpoint recommendation (NOT auto selection)
    if not ckpt_candidates.empty:
        md.append("## Checkpoint candidates (recommendation only)\n")
        md.append(
            "아래 표는 **체크포인트를 자동으로 선택하지 않고**, "
            "val metric 기준으로 **후보를 정렬/매핑**해서 보여줍니다. "
            "최종 선택은 (1) 최고 성능, (2) plateau/변동성, (3) train loss 대비 val metric 추세를 함께 보고 결정하는 것을 권장합니다.\n"
        )
        md.append(df_to_markdown_table(ckpt_candidates.head(args.topk_ckpt), max_rows=args.topk_ckpt))
        md.append("")
        if ckpt_notes:
            md.append("### Notes\n")
            for n in ckpt_notes:
                md.append(f"- {n}")
            md.append("")

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print("Saved files in:", out_dir)
    for f in [
        "train_metrics.png",
        "val_metrics.png",
        "performance.png",
        "loss_breakdown.png",
        "per_class_iou.png",
        "per_class_acc.png",
        "metric_summary.csv",
        "performance_summary.csv",
        "loss_breakdown_summary.csv",
        "checkpoint_candidates.csv",
        "report.md",
    ]:
        p = out_dir / f
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    main()

'''
python log_report.py --log /workspace/projects/vap/work_dirs/fcn_hr48_4xb2-160k_cityscapes-512x1024_vap_large8/20260223_095858/vis_data/20260223_095858.json \
--out /workspace/projects/vap/work_dirs/fcn_hr48_4xb2-160k_cityscapes-512x1024_vap_large8/20260223_095858/vis_data \
--samples_per_iter 8 --primary_metric "mIoU|aAcc|mAcc" --topk_ckpt 3
'''
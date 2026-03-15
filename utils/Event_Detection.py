
from __future__ import annotations

import math
import os
from pathlib import Path
import glob
from typing import Tuple, cast

from numpy.typing import NDArray

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

FEATURES_OUT = "event_detection_features_from_cleaned_quotes.csv"
EVENTS_OUT = "event_windows_summary.csv"
OUTER_MODEL_OUT = "outer_model_coefficients.csv"
OUTER_RESID_SUMMARY_OUT = "outer_residual_regime_summary.csv"
DEFAULT_INPUT_DIR = Path(".")
DEFAULT_OUTPUT_DIR = Path("Event_Det_Out")
INPUT_FILE = "cleaned_quotes_with_iv_1dte.csv"

PLOT_SCORE_PREFIX = "event_score"
PLOT_STATE_PREFIX = "state_series"
EVENT_PLOT_DIR = "images"

ATM_K_ABS_MAX = 0.015
ATM_TOP_N = 6
SHAPE_CAP = 0.035
ATM_BAND = 0.006
N_LIN_EACH_SIDE = 10

# Event detection: stricter + smoother
EVENT_HIGH_Q = 0.975
EVENT_LOW_Q = 0.85
START_CONSEC = 2
END_CONSEC = 2
SMOOTH_WINDOW = 5
MIN_EVENT_LEN = 3
MERGE_GAP = 2

# Plot control
MAX_PLOTS = 7
PLOT_SELECTION_COL = "max_event_score"

EPS = 1e-12


# ============================================================
# Output helpers
# ============================================================
def _out_path(filename: str, output_dir: Path) -> str:
    return str(output_dir / filename)


def cleanup_local_event_detection_outputs(output_dir: Path, legacy_source_dir: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / EVENT_PLOT_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    source_dir = legacy_source_dir or Path.cwd()

    legacy_files = [
        Path(FEATURES_OUT).name,
        Path(EVENTS_OUT).name,
        Path(OUTER_MODEL_OUT).name,
        Path(OUTER_RESID_SUMMARY_OUT).name,
    ]

    for name in legacy_files:
        src = source_dir / name
        if src.is_file():
            dst = _out_path(name, output_dir=output_dir)
            os.replace(src, dst)

    for pattern in (f"{PLOT_SCORE_PREFIX}_*.png", f"{PLOT_STATE_PREFIX}_*.png"):
        for src in source_dir.glob(pattern):
            dst = str(images_dir / src.name)
            os.replace(src, dst)


# Helpers reused from your surface code
# ============================================================
def weighted_slope_fixed_intercept(
    k: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    intercept: float,
) -> float:
    k = np.asarray(k, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    m = np.isfinite(k) & np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(intercept)
    k, y, w = k[m], y[m], w[m]
    if len(k) < 4:
        return float("nan")

    num = np.sum(w * k * (y - intercept))
    den = np.sum(w * k * k)
    if den <= 0:
        return float("nan")
    return float(num / den)


def shape_weights_from_df(df: pd.DataFrame) -> np.ndarray:
    vega = df["vega"].clip(lower=0.0).fillna(0.0).values
    vol = np.sqrt(df["volume"].clip(lower=1.0).fillna(1.0).values)
    raw = vega * vol
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 0.0)

    if np.any(raw > 0):
        raw = np.clip(raw, 0.0, np.nanpercentile(raw[raw > 0], 95))

    return np.clip(raw, 0.0, 1e6)


def collapse_to_strikes(g: pd.DataFrame) -> pd.DataFrame:
    if g.empty:
        return g.copy()

    gg = g.copy()
    gg["is_otm_pref"] = (
        (((gg["cp"] == "P") & (gg["K"] <= gg["S_used"])) |
         ((gg["cp"] == "C") & (gg["K"] >= gg["S_used"])))
    ).astype(int)

    gg["abs_k"] = np.abs(gg["k"])
    gg["volume_rank"] = gg["volume"].fillna(0.0)

    gg = (
        gg.sort_values(
            ["K", "is_otm_pref", "abs_k", "volume_rank", "timestamp"],
            ascending=[True, False, True, False, False],
        )
        .drop_duplicates(subset=["K"], keep="first")
        .copy()
    )
    return gg


def atm_iv_from_nearby(g_loc: pd.DataFrame, k_abs_max: float = 0.015, top_n: int = 6) -> float:
    gg = g_loc.copy()
    gg = gg[np.isfinite(gg["iv"]) & (gg["iv"] > 0)].copy()
    if gg.empty:
        return float("nan")

    gg = collapse_to_strikes(gg)
    gg["abs_k"] = np.abs(gg["k"])
    gg = gg[gg["abs_k"] <= k_abs_max].sort_values("abs_k").head(top_n).copy()

    if gg.empty:
        return float("nan")

    w = shape_weights_from_df(gg).astype(float)
    iv_values = gg["iv"].to_numpy(dtype=float)
    if np.sum(w) <= EPS:
        return float(np.nanmedian(iv_values))

    return float(np.sum(w * iv_values) / np.sum(w))


def build_sigma_skew_features_from_quotes(q: pd.DataFrame) -> pd.DataFrame:
    feats = []

    groups = list(q.groupby(["expiry_date", "tbin"]))
    print(f"[INFO] Cross-sections to process: {len(groups):,}")

    for (exp, tb), g_now in groups:
        if g_now.empty:
            continue

        tau_med = float(np.nanmedian(g_now["tau"].to_numpy(dtype=float))) if not g_now.empty else float("nan")

        g = g_now[np.isfinite(g_now["iv"]) & np.isfinite(g_now["S_used"]) & (g_now["S_used"] > 0)].copy()
        if g.empty:
            continue

        g["k"] = np.log(g["K"] / g["S_used"])
        g["abs_k"] = np.abs(g["k"])
        g = g[np.isfinite(g["k"])].copy()
        g = g[(g["iv"] > 0) & (g["iv"] <= 2.5)].copy()
        if g.empty:
            continue

        g = collapse_to_strikes(g)

        sigma_atm = atm_iv_from_nearby(g, k_abs_max=ATM_K_ABS_MAX, top_n=ATM_TOP_N)
        if not (np.isfinite(sigma_atm) and sigma_atm > 0):
            continue

        g_shape = g[g["abs_k"] <= SHAPE_CAP].copy()
        if g_shape.empty:
            feats.append(
                {
                    "timestamp": tb,
                    "expiry": exp,
                    "tau": tau_med,
                    "dte_bucket": "1DTE",
                    "sigma_atm": float(sigma_atm),
                    "skew": float("nan"),
                    "num_quotes": 0,
                    "shape_ok": False,
                }
            )
            continue

        g_left = g_shape[g_shape["k"] < 0].sort_values("abs_k").copy()
        g_right = g_shape[g_shape["k"] > 0].sort_values("abs_k").copy()
        g_atm = g_shape[g_shape["abs_k"] <= ATM_BAND].sort_values("abs_k").head(2).copy()

        g_lin = (
            pd.concat(
                [
                    g_left.head(N_LIN_EACH_SIDE),
                    g_right.head(N_LIN_EACH_SIDE),
                    g_atm,
                ],
                ignore_index=True,
            )
            .sort_values("abs_k")
            .drop_duplicates(subset=["K"], keep="first")
            .copy()
        )

        left_unique = g_lin.loc[g_lin["k"] < 0, "K"].nunique()
        right_unique = g_lin.loc[g_lin["k"] > 0, "K"].nunique()
        total_unique = g_lin["K"].nunique()
        left_span = abs(g_lin.loc[g_lin["k"] < 0, "k"].min()) if (g_lin["k"] < 0).any() else 0.0
        right_span = g_lin.loc[g_lin["k"] > 0, "k"].max() if (g_lin["k"] > 0).any() else 0.0

        skew_ok = (
            total_unique >= 6
            and left_unique >= 2
            and right_unique >= 2
            and left_span >= 0.0025
            and right_span >= 0.0025
        )

        skew = float("nan")
        if skew_ok:
            w_lin = shape_weights_from_df(g_lin)
            b_lin = weighted_slope_fixed_intercept(
                g_lin["k"].to_numpy(dtype=float),
                g_lin["iv"].to_numpy(dtype=float),
                w_lin,
                sigma_atm,
            )
            if np.isfinite(b_lin):
                skew = float(b_lin)

        feats.append(
            {
                "timestamp": tb,
                "expiry": exp,
                "tau": tau_med,
                "dte_bucket": "1DTE",
                "sigma_atm": float(sigma_atm),
                "skew": float(skew),
                "num_quotes": int(len(g_shape)),
                "shape_ok": bool(np.isfinite(skew)),
            }
        )

    feat = pd.DataFrame(feats).sort_values(["expiry", "timestamp"]).reset_index(drop=True)
    feat["date"] = pd.to_datetime(feat["timestamp"]).dt.date
    return feat


# ============================================================
# Event detection
# ============================================================
def robust_zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-10:
        std = x.std()
        if not np.isfinite(std) or std < 1e-10:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.mean()) / std
    return (x - med) / scale


def add_daily_dynamics(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    out = out.sort_values(["expiry", "timestamp"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["timestamp"]).dt.date

    def _per_expiry_day(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.sort_values("timestamp").copy()
        gg["d_sigma"] = gg["sigma_atm"].diff()
        gg["d_skew"] = gg["skew"].diff()
        gg["dd_sigma"] = gg["d_sigma"].diff()
        gg["dd_skew"] = gg["d_skew"].diff()

        gg["z_d_sigma"] = robust_zscore(gg["d_sigma"])
        gg["z_d_skew"] = robust_zscore(gg["d_skew"])
        gg["z_dd_sigma"] = robust_zscore(gg["dd_sigma"])
        gg["z_dd_skew"] = robust_zscore(gg["dd_skew"])

        gg["event_score"] = np.sqrt(
            gg["z_d_sigma"].fillna(0.0) ** 2 +
            gg["z_d_skew"].fillna(0.0) ** 2
        )

        gg["event_score_plus"] = np.sqrt(
            gg["z_d_sigma"].fillna(0.0) ** 2 +
            gg["z_d_skew"].fillna(0.0) ** 2 +
            0.25 * gg["z_dd_sigma"].fillna(0.0) ** 2 +
            0.25 * gg["z_dd_skew"].fillna(0.0) ** 2
        )

        gg["event_score_smooth"] = (
            gg["event_score"]
            .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
            .median()
        )
        return gg

    out = (
        out.groupby(["expiry", "date"], group_keys=True)
        .apply(_per_expiry_day)
    )
    if isinstance(out.index, pd.MultiIndex):
        level_cols = [c for c in ["level_0", "level_1", "level_2", "level_3"] if c in out.columns]
        if level_cols:
            out = out.drop(columns=level_cols)
        out = out.reset_index()
        if "level_0" in out.columns:
            out = out.rename(columns={"level_0": "expiry"})
        if "level_1" in out.columns:
            out = out.rename(columns={"level_1": "date"})
    else:
        out = out.reset_index(drop=True)

    if "expiry" not in out.columns or "date" not in out.columns:
        raise KeyError(
            "add_daily_dynamics failed to retain required columns. "
            f"Got columns: {list(out.columns)}"
        )

    out = out.sort_values(["expiry", "timestamp"]).reset_index(drop=True)
    return out


def hysteresis_event_labels(
    score: pd.Series,
    high_q: float = 0.975,
    low_q: float = 0.85,
    start_consec: int = 2,
    end_consec: int = 2,
) -> pd.Series:
    x = pd.to_numeric(score, errors="coerce").fillna(0.0).values
    if len(x) == 0:
        return pd.Series(dtype=int)

    x_series = pd.Series(x.astype(float))
    high_thr = float(x_series.quantile(high_q))
    low_thr = float(x_series.quantile(low_q))

    labels = np.zeros(len(x), dtype=int)
    state = 0
    high_run = 0
    low_run = 0

    for i, val in enumerate(x):
        if state == 0:
            if val > high_thr:
                high_run += 1
            else:
                high_run = 0

            if high_run >= start_consec:
                state = 1
                start_idx = i - start_consec + 1
                labels[start_idx:i + 1] = 1
                low_run = 0
        else:
            labels[i] = 1
            if val < low_thr:
                low_run += 1
            else:
                low_run = 0

            if low_run >= end_consec:
                end_back = i - end_consec + 1
                labels[end_back:i + 1] = 0
                state = 0
                high_run = 0
                low_run = 0

    return pd.Series(labels, index=score.index)


def drop_short_events(is_event: pd.Series, min_len: int = 3) -> pd.Series:
    x = is_event.astype(int).copy()
    event_start = (x.eq(1) & x.shift(fill_value=0).eq(0)).cumsum()
    block_ids = pd.Series(np.where(x.eq(1), event_start, np.nan), index=x.index)

    for _, idx in block_ids.dropna().groupby(block_ids.dropna()).groups.items():
        idx = list(idx)
        if len(idx) < min_len:
            x.loc[idx] = 0
    return x


def merge_nearby_events(is_event: pd.Series, max_gap: int = 2) -> pd.Series:
    x = is_event.astype(int).copy()
    vals = x.values.copy()
    n = len(vals)

    i = 0
    while i < n:
        if vals[i] == 1:
            j = i
            while j < n and vals[j] == 1:
                j += 1
            k = j
            while k < n and vals[k] == 0:
                k += 1
            if k < n and (k - j) <= max_gap:
                vals[j:k] = 1
            i = k
        else:
            i += 1

    return pd.Series(vals, index=x.index)


def label_events(feat_dyn: pd.DataFrame, score_col: str = "event_score_smooth") -> pd.DataFrame:
    out = feat_dyn.copy()
    if "expiry" not in out.columns and "expiry_date" in out.columns:
        out = out.rename(columns={"expiry_date": "expiry"})
    if "date" not in out.columns and "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"]).dt.date

    if "expiry" not in out.columns or "date" not in out.columns:
        raise KeyError(
            "label_events expects both 'expiry' and 'date' columns. "
            f"Got columns: {list(out.columns)}"
        )

    def _per_expiry_day(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.sort_values("timestamp").copy()
        gg["is_event"] = hysteresis_event_labels(
            gg[score_col],
            high_q=EVENT_HIGH_Q,
            low_q=EVENT_LOW_Q,
            start_consec=START_CONSEC,
            end_consec=END_CONSEC,
        )
        gg["is_event"] = drop_short_events(gg["is_event"], min_len=MIN_EVENT_LEN)
        gg["is_event"] = merge_nearby_events(gg["is_event"], max_gap=MERGE_GAP)
        gg["is_event"] = drop_short_events(gg["is_event"], min_len=MIN_EVENT_LEN)
        return gg

    try:
        out = (
            out.groupby(["expiry", "date"], group_keys=True)
            .apply(_per_expiry_day)
        )
        if isinstance(out.index, pd.MultiIndex):
            level_cols = [c for c in ["level_0", "level_1", "level_2", "level_3"] if c in out.columns]
            if level_cols:
                out = out.drop(columns=level_cols)
            out = out.reset_index()
            if "level_0" in out.columns:
                out = out.rename(columns={"level_0": "expiry"})
            if "level_1" in out.columns:
                out = out.rename(columns={"level_1": "date"})
        else:
            out = out.reset_index(drop=True)
    except Exception as e:
        raise e

    if "expiry" not in out.columns or "date" not in out.columns:
        raise KeyError(
            "label_events failed to retain required columns after event labeling. "
            f"Got columns: {list(out.columns)}"
        )
    
    return out


def summarize_event_windows(feat_evt: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (exp, day), g in feat_evt.groupby(["expiry", "date"]):
        gg = g.sort_values("timestamp").copy()

        event_block = []
        block_id = 0
        prev = 0
        for val in gg["is_event"].astype(int).values:
            if val == 1 and prev == 0:
                block_id += 1
            event_block.append(block_id if val == 1 else np.nan)
            prev = val
        gg["event_block"] = event_block

        for i, (_, w) in enumerate(
            gg.dropna(subset=["event_block"]).groupby("event_block"),
            start=1,
        ):
            rows.append(
                {
                    "expiry": exp,
                    "date": day,
                    "event_id": i,
                    "start_time": w["timestamp"].min(),
                    "end_time": w["timestamp"].max(),
                    "minutes": int(len(w)),
                    "max_event_score": float(w["event_score"].max()),
                    "max_event_score_smooth": float(w["event_score_smooth"].max()),
                    "mean_event_score": float(w["event_score"].mean()),
                    "max_abs_d_sigma": float(np.nanmax(np.abs(w["d_sigma"].values))),
                    "max_abs_d_skew": float(np.nanmax(np.abs(w["d_skew"].values))),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "expiry", "date", "event_id", "start_time", "end_time", "minutes",
                "max_event_score", "max_event_score_smooth", "mean_event_score",
                "max_abs_d_sigma", "max_abs_d_skew",
            ]
        )

    return pd.DataFrame(rows).sort_values(["date", "expiry", "start_time"]).reset_index(drop=True)


# ============================================================
# Calm-regime outer model
# ============================================================
def fit_outer_var1(feat_evt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["timestamp", "expiry", "date", "sigma_atm", "skew", "is_event"]
    d = feat_evt[cols].copy()
    d = d.sort_values(["expiry", "timestamp"]).reset_index(drop=True)

    frames = []
    coef_rows = []

    for exp, g in d.groupby("expiry"):
        gg = g.sort_values("timestamp").copy()

        gg["sigma_next"] = gg["sigma_atm"].shift(-1)
        gg["skew_next"] = gg["skew"].shift(-1)

        calm = gg[
            (gg["is_event"] == 0) &
            (gg["sigma_atm"].notna()) &
            (gg["skew"].notna()) &
            (gg["sigma_next"].notna()) &
            (gg["skew_next"].notna())
        ].copy()

        if len(calm) < 20:
            continue

        X: NDArray[np.float64] = np.column_stack([
            np.ones(len(calm), dtype=float),
            calm["sigma_atm"].to_numpy(dtype=float),
            calm["skew"].to_numpy(dtype=float),
        ])

        y_sigma = calm["sigma_next"].to_numpy(dtype=float)
        y_skew = calm["skew_next"].to_numpy(dtype=float)

        beta_sigma, *_ = np.linalg.lstsq(X, y_sigma, rcond=None)
        beta_skew, *_ = np.linalg.lstsq(X, y_skew, rcond=None)

        coef_rows.extend(
            [
                {
                    "expiry": exp,
                    "equation": "sigma_next",
                    "intercept": float(beta_sigma[0]),
                    "coef_sigma": float(beta_sigma[1]),
                    "coef_skew": float(beta_sigma[2]),
                    "n_obs": int(len(calm)),
                },
                {
                    "expiry": exp,
                    "equation": "skew_next",
                    "intercept": float(beta_skew[0]),
                    "coef_sigma": float(beta_skew[1]),
                    "coef_skew": float(beta_skew[2]),
                    "n_obs": int(len(calm)),
                },
            ]
        )

        X_all: NDArray[np.float64] = np.column_stack([
            np.ones(len(gg), dtype=float),
            gg["sigma_atm"].ffill().bfill().to_numpy(dtype=float),
            gg["skew"].ffill().bfill().to_numpy(dtype=float),
        ])

        gg["sigma_pred_outer"] = X_all @ beta_sigma
        gg["skew_pred_outer"] = X_all @ beta_skew

        gg["sigma_resid_outer"] = gg["sigma_next"] - gg["sigma_pred_outer"]
        gg["skew_resid_outer"] = gg["skew_next"] - gg["skew_pred_outer"]

        gg["outer_resid_norm"] = np.sqrt(
            gg["sigma_resid_outer"].fillna(0.0) ** 2 +
            gg["skew_resid_outer"].fillna(0.0) ** 2
        )
        frames.append(gg)

    coef_df = pd.DataFrame(coef_rows)
    fitted_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return coef_df, fitted_df


def summarize_outer_residuals(outer_fit: pd.DataFrame) -> pd.DataFrame:
    if outer_fit.empty:
        return pd.DataFrame(
            columns=[
                "expiry", "date", "event_state", "rows",
                "mean_outer_resid_norm", "median_outer_resid_norm", "p95_outer_resid_norm",
                "mean_abs_sigma_resid", "mean_abs_skew_resid",
            ]
        )

    d = outer_fit.copy()
    d["abs_sigma_resid"] = d["sigma_resid_outer"].abs()
    d["abs_skew_resid"] = d["skew_resid_outer"].abs()
    d["event_state"] = np.where(d["is_event"] == 1, "event", "calm")

    out = (
        d.groupby(["expiry", "date", "event_state"])
        .agg(
            rows=("timestamp", "size"),
            mean_outer_resid_norm=("outer_resid_norm", "mean"),
            median_outer_resid_norm=("outer_resid_norm", "median"),
            p95_outer_resid_norm=(
                "outer_resid_norm",
                lambda s: float(np.nanquantile(s.to_numpy(dtype=float), 0.95)),
            ),
            mean_abs_sigma_resid=("abs_sigma_resid", "mean"),
            mean_abs_skew_resid=("abs_skew_resid", "mean"),
        )
        .reset_index()
        .sort_values(["date", "expiry", "event_state"])
        .reset_index(drop=True)
    )
    return out


# ============================================================
# Plot helpers
# ============================================================
def choose_plot_groups(feat_evt: pd.DataFrame, max_plots: int = 7) -> list[tuple]:
    summary = (
        feat_evt.groupby(["expiry", "date"])
        .agg(
            max_event_score=("event_score", "max"),
            max_event_score_smooth=("event_score_smooth", "max"),
            event_rows=("is_event", "sum"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        [PLOT_SELECTION_COL, "event_rows"],
        ascending=[False, False],
    ).head(max_plots)

    return [(row["expiry"], row["date"]) for _, row in summary.iterrows()]


def shade_event_windows(ax, gg: pd.DataFrame) -> None:
    in_event = False
    start_ts: pd.Timestamp | None = None
    ts_series = pd.to_datetime(gg["timestamp"])
    for _, row in gg.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        if int(row["is_event"]) == 1 and not in_event:
            in_event = True
            start_ts = ts
        elif int(row["is_event"]) == 0 and in_event:
            ax.axvspan(cast(pd.Timestamp, start_ts), ts, alpha=0.2)
            in_event = False
            start_ts = None
    if in_event and start_ts is not None:
        ax.axvspan(cast(pd.Timestamp, start_ts), ts_series.iloc[-1], alpha=0.2)


def plot_selected_event_scores(feat_evt: pd.DataFrame, selected_groups: list[tuple], prefix: str) -> None:
    for exp, day in selected_groups:
        gg = feat_evt[(feat_evt["expiry"] == exp) & (feat_evt["date"] == day)].sort_values("timestamp").copy()
        if gg.empty:
            continue

        plt.figure(figsize=(12, 4))
        plt.plot(pd.to_datetime(gg["timestamp"]), gg["event_score"], linewidth=1.2, label="raw score")
        plt.plot(pd.to_datetime(gg["timestamp"]), gg["event_score_smooth"], linewidth=2.0, label="smoothed score")
        shade_event_windows(plt.gca(), gg)

        plt.title(f"Event score | expiry {pd.to_datetime(exp).date()} | trade date {day}")
        plt.xlabel("Time")
        plt.ylabel("event_score")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{prefix}_{pd.to_datetime(exp).date()}_{day}.png", dpi=140)
        plt.close()


def plot_selected_state_series(feat_evt: pd.DataFrame, selected_groups: list[tuple], prefix: str) -> None:
    for exp, day in selected_groups:
        gg = feat_evt[(feat_evt["expiry"] == exp) & (feat_evt["date"] == day)].sort_values("timestamp").copy()
        if gg.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axes[0].plot(pd.to_datetime(gg["timestamp"]), gg["sigma_atm"], linewidth=1.5)
        axes[0].set_ylabel("sigma_atm")
        axes[0].set_title(f"State series | expiry {pd.to_datetime(exp).date()} | trade date {day}")

        axes[1].plot(pd.to_datetime(gg["timestamp"]), gg["skew"], linewidth=1.5)
        axes[1].set_ylabel("skew")
        axes[1].set_xlabel("Time")

        for ax in axes:
            shade_event_windows(ax, gg)

        plt.tight_layout()
        plt.savefig(f"{prefix}_{pd.to_datetime(exp).date()}_{day}.png", dpi=140)
        plt.close()


# ============================================================
# Main
# ============================================================
def main(
    INPUT_DIR: str = str(DEFAULT_INPUT_DIR),
    INPUT_FILE: str = INPUT_FILE,
    OUTPUT_DIR: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    input_dir = Path(INPUT_DIR).expanduser()
    output_dir = Path(OUTPUT_DIR).expanduser()
    input_file = input_dir / INPUT_FILE

    cleanup_local_event_detection_outputs(output_dir=output_dir, legacy_source_dir=input_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Expected input file: {input_file}")

    q = pd.read_csv(input_file)

    datetime_cols = ["timestamp", "expiry_date", "trade_date", "tbin", "quote_timestamp"]
    for c in datetime_cols:
        if c in q.columns:
            q[c] = pd.to_datetime(q[c])

    numeric_cols = [
        "K", "open", "close", "value", "volume", "last", "mid",
        "tau_min", "tau", "S_used", "r", "iv", "vega", "k",
        "days_to_expiry", "quote_age_sec", "model_price",
        "abs_price_err", "rel_price_err",
    ]
    for c in numeric_cols:
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q[q["dte_bucket"] == "1DTE"].copy()
    q = q[np.isfinite(q["iv"]) & np.isfinite(q["vega"]) & np.isfinite(q["S_used"])].copy()

    print(f"[INFO] Loaded rows: {len(q):,}")
    print(f"[INFO] Time range: {q['timestamp'].min()} to {q['timestamp'].max()}")

    feat = build_sigma_skew_features_from_quotes(q)
    print(f"[INFO] Feature rows: {len(feat):,}")
    print(feat.columns)

    feat_dyn = add_daily_dynamics(feat)
    if "expiry" not in feat_dyn.columns and "expiry_date" in feat_dyn.columns:
        feat_dyn = feat_dyn.rename(columns={"expiry_date": "expiry"})
    if "date" not in feat_dyn.columns and "trade_date" in feat.columns:
        feat_dyn["date"] = pd.to_datetime(feat["trade_date"]).dt.date
    if "expiry" not in feat_dyn.columns or "date" not in feat_dyn.columns:
        raise KeyError(
            "main expects both 'expiry' and 'date' columns after dynamic feature creation. "
            f"Got columns: {list(feat_dyn.columns)}"
        )
    feat_evt = label_events(feat_dyn, score_col="event_score_smooth")

    events_summary = summarize_event_windows(feat_evt)
    coef_df, outer_fit = fit_outer_var1(feat_evt)
    outer_resid_summary = summarize_outer_residuals(outer_fit)

    feat_evt.to_csv(_out_path(Path(FEATURES_OUT).name, output_dir=output_dir), index=False)
    events_summary.to_csv(_out_path(Path(EVENTS_OUT).name, output_dir=output_dir), index=False)
    coef_df.to_csv(_out_path(Path(OUTER_MODEL_OUT).name, output_dir=output_dir), index=False)
    outer_resid_summary.to_csv(_out_path(Path(OUTER_RESID_SUMMARY_OUT).name, output_dir=output_dir), index=False)

    selected_groups = choose_plot_groups(feat_evt, max_plots=MAX_PLOTS)
    plot_output_dir = output_dir / EVENT_PLOT_DIR
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_selected_event_scores(
        feat_evt,
        selected_groups,
        _out_path(PLOT_SCORE_PREFIX, output_dir=plot_output_dir),
    )
    plot_selected_state_series(
        feat_evt,
        selected_groups,
        _out_path(PLOT_STATE_PREFIX, output_dir=plot_output_dir),
    )

    print("\n[INFO] Files written:")
    print(f"  {_out_path(Path(FEATURES_OUT).name, output_dir=output_dir)}")
    print(f"  {_out_path(Path(EVENTS_OUT).name, output_dir=output_dir)}")
    print(f"  {_out_path(Path(OUTER_MODEL_OUT).name, output_dir=output_dir)}")
    print(f"  {_out_path(Path(OUTER_RESID_SUMMARY_OUT).name, output_dir=output_dir)}")

    print(f"\n[INFO] Plots written for up to {MAX_PLOTS} expiry/day groups:")
    for exp, day in selected_groups:
        print(f"  expiry={pd.to_datetime(exp).date()} | trade_date={day}")

    print("\n[INFO] Event coverage summary:")
    tmp = feat_evt.groupby(["date", "expiry"]).agg(
        rows=("timestamp", "size"),
        event_rows=("is_event", "sum"),
        mean_score=("event_score", "mean"),
        max_score=("event_score", "max"),
        max_score_smooth=("event_score_smooth", "max"),
    ).reset_index()
    print(tmp.to_string(index=False))

    print("\n[INFO] Outer model coefficients:")
    if not coef_df.empty:
        print(coef_df.to_string(index=False))
    else:
        print("No calm-regime model fit was produced.")

    print("\n[INFO] Outer residual summary:")
    if not outer_resid_summary.empty:
        print(outer_resid_summary.to_string(index=False))
    else:
        print("No outer residual summary produced.")


if __name__ == "__main__":
    main()

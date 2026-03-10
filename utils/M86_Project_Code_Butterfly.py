

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_INPUT_DIR = Path("garbage")
DEFAULT_OUTPUT_DIR = Path("garbage2")
IV_INPUT_FILE = "cleaned_quotes_with_iv_1dte.csv"

# -----------------------------
# Tunable parameters
# -----------------------------
H = 0.01                  # wing location in log-moneyness
TOL = 0.0025              # neighborhood tolerance around -H, 0, +H
EWM_SPAN = 7              # smoothing span inside each day
CUSUM_KAPPA = 0.5         # reference value in z-score units
CUSUM_H = 6.0             # alarm threshold in z-score units
EVENT_GAP_MIN = 15        # minimum minutes between stored event snapshots
TRADE_DATE = None         # e.g. "2026-02-03"; if None, auto-pick top event day

# -----------------------------
# Helpers
# -----------------------------
def zscore_day(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad
    if (not np.isfinite(scale)) or (scale < 1e-8):
        scale = s.std()
    if (not np.isfinite(scale)) or (scale < 1e-8):
        scale = 1.0
    return (s - med) / scale


def cusum_series(s: pd.Series, kappa: float = CUSUM_KAPPA, h: float = CUSUM_H) -> pd.DataFrame:
    z = s.to_numpy(dtype=float)
    pos = np.zeros(len(z))
    neg = np.zeros(len(z))
    event = np.zeros(len(z), dtype=int)
    direction = np.zeros(len(z), dtype=int)

    for i in range(1, len(z)):
        zi = z[i]
        if not np.isfinite(zi):
            continue

        pos[i] = max(0.0, pos[i - 1] + zi - kappa)
        neg[i] = min(0.0, neg[i - 1] + zi + kappa)

        if pos[i] > h:
            event[i] = 1
            direction[i] = 1
            pos[i] = 0.0
            neg[i] = 0.0
        elif neg[i] < -h:
            event[i] = 1
            direction[i] = -1
            pos[i] = 0.0
            neg[i] = 0.0

    return pd.DataFrame(
        {"cusum_pos": pos, "cusum_neg": neg, "event": event, "direction": direction},
        index=s.index,
    )


def local_sigma_from_bin(x: np.ndarray, y: np.ndarray, target: float, tol: float = TOL, min_pts: int = 3) -> float:
    dist = np.abs(x - target)
    idx = np.where(dist <= tol)[0]
    if idx.size < min_pts:
        idx = np.argsort(dist)[:min_pts]
    return float(np.median(y[idx]))


# -----------------------------
# Feature extraction
# -----------------------------
def build_features(csv_path: Path, h: float = H, tol: float = TOL) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["timestamp", "trade_date", "k", "iv", "vega"]
    q = pd.read_csv(csv_path, usecols=usecols)

    q["timestamp"] = pd.to_datetime(q["timestamp"])
    q["trade_date"] = pd.to_datetime(q["trade_date"]).dt.normalize()

    q = q.dropna(subset=["timestamp", "trade_date", "k", "iv", "vega"]).copy()
    q = q[np.isfinite(q["k"]) & np.isfinite(q["iv"]) & np.isfinite(q["vega"])].copy()
    q = q[(q["iv"] > 0.0) & (q["iv"] < 3.0)].copy()

    # Market hours only
    open_time = pd.Timestamp("09:30").time()
    close_time = pd.Timestamp("16:00").time()
    q = q[(q["timestamp"].dt.time >= open_time) & (q["timestamp"].dt.time <= close_time)].copy()

    # Vega-weighted collapse to one IV per (timestamp, k_bin)
    q["w"] = q["vega"].clip(lower=1e-8)
    q["k_bin"] = q["k"].round(4)
    q["k_w"] = q["k"] * q["w"]
    q["iv_w"] = q["iv"] * q["w"]

    agg = (
        q.groupby(["timestamp", "trade_date", "k_bin"], as_index=False)
         .agg(w=("w", "sum"), k_w=("k_w", "sum"), iv_w=("iv_w", "sum"), n=("w", "size"))
    )
    agg["k"] = agg["k_w"] / agg["w"]
    agg["iv"] = agg["iv_w"] / agg["w"]
    agg = agg[["timestamp", "trade_date", "k", "iv", "w", "n"]].sort_values(["timestamp", "k"]).reset_index(drop=True)

    rows = []
    for ts, grp in agg.groupby("timestamp", sort=True):
        x = grp["k"].to_numpy(dtype=float)
        y = grp["iv"].to_numpy(dtype=float)

        if len(x) < 8:
            continue

        sigma_left = local_sigma_from_bin(x, y, -h, tol=tol)
        sigma_atm = local_sigma_from_bin(x, y, 0.0, tol=tol)
        sigma_right = local_sigma_from_bin(x, y, h, tol=tol)

        butterfly = 0.5 * (sigma_left + sigma_right) - sigma_atm
        curvature_bfly = 2.0 * butterfly / (h ** 2)
        skew = (sigma_right - sigma_left) / (2.0 * h)

        rows.append(
            {
                "timestamp": ts,
                "trade_date": grp["trade_date"].iloc[0],
                "sigma_atm": sigma_atm,
                "skew": skew,
                "bfly": butterfly,
                "curvature_bfly": curvature_bfly,
                "sigma_left": sigma_left,
                "sigma_right": sigma_right,
                "n_pts": len(grp),
            }
        )

    feat = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # Intraday-only detector: each day is independent
    for col in ["sigma_atm", "skew", "curvature_bfly"]:
        feat[f"{col}_sm"] = feat.groupby("trade_date")[col].transform(
            lambda s: s.ewm(span=EWM_SPAN, adjust=False).mean()
        )
        feat[f"d_{col}"] = feat.groupby("trade_date")[f"{col}_sm"].diff()
        feat[f"z_{col}"] = feat.groupby("trade_date")[f"d_{col}"].transform(zscore_day).clip(-10, 10)

        tmp = feat.groupby("trade_date")[f"z_{col}"].apply(cusum_series).reset_index(level=0, drop=True)
        feat[f"cusum_pos_{col}"] = tmp["cusum_pos"]
        feat[f"cusum_neg_{col}"] = tmp["cusum_neg"]
        feat[f"event_{col}"] = tmp["event"]
        feat[f"dir_{col}"] = tmp["direction"]

    feat["event_any"] = feat[[f"event_{c}" for c in ["sigma_atm", "skew", "curvature_bfly"]]].max(axis=1)
    feat["event_score"] = feat[[f"z_{c}" for c in ["sigma_atm", "skew", "curvature_bfly"]]].abs().sum(axis=1)

    day_summary = (
        feat.groupby("trade_date", as_index=False)
        .agg(
            n_obs=("timestamp", "size"),
            n_events=("event_any", "sum"),
            max_event_score=("event_score", "max"),
            sigma_range=("sigma_atm", lambda s: float(s.max() - s.min())),
            skew_range=("skew", lambda s: float(s.max() - s.min())),
            curvature_range=("curvature_bfly", lambda s: float(s.max() - s.min())),
        )
        .sort_values(["max_event_score", "n_events"], ascending=False)
        .reset_index(drop=True)
    )

    return agg, feat, day_summary


# -----------------------------
# Plotting
# -----------------------------
def pick_day(feat: pd.DataFrame, requested_day: str | None) -> pd.Timestamp:
    if requested_day is not None:
        day = pd.to_datetime(requested_day).normalize()
        if day not in set(feat["trade_date"]):
            raise ValueError(f"TRADE_DATE={requested_day} not present in data")
        return day

    tmp = (
        feat.groupby("trade_date", as_index=False)
        .agg(max_event_score=("event_score", "max"), n_events=("event_any", "sum"))
        .sort_values(["max_event_score", "n_events"], ascending=False)
        .reset_index(drop=True)
    )
    return tmp.loc[0, "trade_date"]


def intraday_events_for_day(day_df: pd.DataFrame, min_gap_min: int = EVENT_GAP_MIN) -> list[pd.Timestamp]:
    cand = day_df[day_df["event_any"] == 1].sort_values("event_score", ascending=False)["timestamp"].tolist()
    chosen = []
    for ts in cand:
        if not chosen or all(abs((ts - old).total_seconds()) > 60 * min_gap_min for old in chosen):
            chosen.append(ts)
        if len(chosen) == 2:
            break
    return chosen


def save_intraday_feature_plot(day_df: pd.DataFrame, ycol: str, title: str, outfile: Path) -> None:
    events = day_df[day_df["event_any"] == 1]
    plt.figure(figsize=(12, 4.3))
    plt.plot(day_df["timestamp"], day_df[ycol], linewidth=1.5)
    if not events.empty:
        plt.scatter(events["timestamp"], events[ycol], s=18)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ycol)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close()


def save_intraday_score_plot(day_df: pd.DataFrame, outfile: Path) -> None:
    plt.figure(figsize=(12, 4.3))
    plt.plot(day_df["timestamp"], day_df["event_score"], linewidth=1.5)
    alarm = day_df[day_df["event_any"] == 1]
    if not alarm.empty:
        plt.scatter(alarm["timestamp"], alarm["event_score"], s=18)
    plt.title("Intraday combined event score")
    plt.xlabel("Time")
    plt.ylabel("event_score")
    plt.tight_layout()
    plt.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close()


def save_smile_plot(agg: pd.DataFrame, event_ts: pd.Timestamp, pre_minutes: int, outfile: Path) -> None:
    pre_ts = event_ts - pd.Timedelta(minutes=pre_minutes)
    same_day = agg["trade_date"] == event_ts.normalize()

    g0 = agg[same_day & (agg["timestamp"] == pre_ts)][["k", "iv"]].sort_values("k")
    g1 = agg[same_day & (agg["timestamp"] == event_ts)][["k", "iv"]].sort_values("k")

    if g0.empty or g1.empty:
        return

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(g0["k"], g0["iv"], marker="o", markersize=3, linewidth=1, label=f"{pre_ts.strftime('%H:%M')}")
    plt.plot(g1["k"], g1["iv"], marker="o", markersize=3, linewidth=1, label=f"{event_ts.strftime('%H:%M')}")
    plt.axvline(-H, linestyle="--", linewidth=1)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.axvline(H, linestyle="--", linewidth=1)
    plt.xlim(-0.03, 0.03)
    plt.title(f"Smile shift around intraday event: {event_ts.strftime('%Y-%m-%d %H:%M')}")
    plt.xlabel("log-moneyness k")
    plt.ylabel("implied vol")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close()


def main(
    INPUT_DIR: str = str(DEFAULT_INPUT_DIR),
    OUTPUT_DIR: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = input_dir / IV_INPUT_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected IV output file: {csv_path}")

    agg, feat, day_summary = build_features(csv_path)
    feat.to_csv(output_dir / "option_b_intraday_features_events_1dte.csv", index=False)
    day_summary.to_csv(output_dir / "option_b_intraday_day_summary_1dte.csv", index=False)

    day = pick_day(feat, TRADE_DATE)
    day_df = feat[feat["trade_date"] == day].copy().sort_values("timestamp")
    chosen = intraday_events_for_day(day_df)

    save_intraday_feature_plot(day_df, "sigma_atm", f"ATM IV intraday — {day.date()}", output_dir / "intraday_plot_1_sigma.png")
    save_intraday_feature_plot(day_df, "skew", f"Skew intraday — {day.date()}", output_dir / "intraday_plot_2_skew.png")
    save_intraday_feature_plot(day_df, "curvature_bfly", f"Butterfly curvature intraday — {day.date()}", output_dir / "intraday_plot_3_curvature.png")
    save_intraday_score_plot(day_df, output_dir / "intraday_plot_4_score.png")

    if len(chosen) >= 1:
        save_smile_plot(agg, chosen[0], pre_minutes=5, outfile=output_dir / "intraday_plot_5_smile_1.png")
    if len(chosen) >= 2:
        save_smile_plot(agg, chosen[1], pre_minutes=5, outfile=output_dir / "intraday_plot_6_smile_2.png")

    print("Top day summary:")
    print(day_summary.head(10).to_string(index=False))
    print("\nSelected intraday day:", day.date())
    print(day_df[["timestamp", "sigma_atm", "skew", "curvature_bfly", "event_score", "event_any"]].head(15).to_string(index=False))

    if chosen:
        print("\nChosen event timestamps:")
        for ts in chosen:
            print(ts)


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER PATHS
# ============================================================
FEATURES_FILE = "option_b_intraday_features_events_1dte.csv"
DEFAULT_INPUT_DIR = Path("Butterfly_Out")
DEFAULT_OUTPUT_DIR = Path("Regime_Out")

# ============================================================
# TUNING
# ============================================================
FEATURES = ["sigma_atm", "skew", "curvature_bfly"]
HIGH_SCORE = 8.0         # enter fast regime if event_score >= HIGH_SCORE or event_any == 1
LOW_SCORE = 4.0          # exit only after score stays below LOW_SCORE long enough
HOLD_BARS = 10           # keep fast regime active for at least this many 1-minute bars
MIN_OBS_PER_REGIME = 25  # minimum transitions needed to fit a regime
USE_EVENT_ANY_TRIGGER = True
PLOT_DAY = None          # e.g. "2026-01-28" ; None = auto-pick best event day


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df = df.sort_values(["trade_date", "timestamp"]).reset_index(drop=True)
    return df



def build_regime_labels(
    df: pd.DataFrame,
    high_score: float = HIGH_SCORE,
    low_score: float = LOW_SCORE,
    hold_bars: int = HOLD_BARS,
    use_event_any_trigger: bool = USE_EVENT_ANY_TRIGGER,
) -> pd.DataFrame:
    out = df.copy()
    out["regime"] = 0
    out["regime_name"] = "slow"
    out["fast_trigger"] = 0

    pieces = []
    for day, g in out.groupby("trade_date", sort=True):
        g = g.sort_values("timestamp").copy()
        regime = np.zeros(len(g), dtype=int)
        trigger = np.zeros(len(g), dtype=int)
        fast = 0
        cooldown = 0

        score = g["event_score"].fillna(0.0).to_numpy(dtype=float)
        event_any = g.get("event_any", pd.Series(0, index=g.index)).fillna(0).to_numpy(dtype=int)

        for i in range(len(g)):
            enter_fast = score[i] >= high_score
            if use_event_any_trigger:
                enter_fast = enter_fast or (event_any[i] == 1)

            if fast == 0:
                if enter_fast:
                    fast = 1
                    cooldown = hold_bars
                    trigger[i] = 1
            else:
                if score[i] >= low_score:
                    cooldown = hold_bars
                else:
                    cooldown -= 1
                    if cooldown <= 0:
                        fast = 0
                        cooldown = 0

            regime[i] = fast

        g["regime"] = regime
        g["regime_name"] = np.where(g["regime"] == 1, "fast", "slow")
        g["fast_trigger"] = trigger
        pieces.append(g)

    out = pd.concat(pieces, axis=0).sort_values(["trade_date", "timestamp"]).reset_index(drop=True)
    return out



def add_lags(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        out[f"{col}_lag1"] = out.groupby("trade_date")[col].shift(1)
        out[f"{col}_lead0"] = out[col]
        out[f"d_{col}"] = out[col] - out[f"{col}_lag1"]

    out["regime_lag1"] = out.groupby("trade_date")["regime"].shift(1)
    out["timestamp_lag1"] = out.groupby("trade_date")["timestamp"].shift(1)
    out["dt_minutes"] = (out["timestamp"] - out["timestamp_lag1"]).dt.total_seconds() / 60.0

    # Keep only true within-day one-step transitions.
    out = out[out["dt_minutes"].between(0.5, 1.5, inclusive="both")].copy()
    return out



def fit_ar1_single_regime(g: pd.DataFrame, feature: str) -> Dict[str, float]:
    x_prev = g[f"{feature}_lag1"].to_numpy(dtype=float)
    x_now = g[feature].to_numpy(dtype=float)

    mask = np.isfinite(x_prev) & np.isfinite(x_now)
    x_prev = x_prev[mask]
    x_now = x_now[mask]

    n = len(x_prev)
    if n < MIN_OBS_PER_REGIME:
        return {
            "n_obs": n,
            "alpha": np.nan,
            "beta": np.nan,
            "mu": np.nan,
            "lambda_per_min": np.nan,
            "half_life_min": np.nan,
            "shock_std": np.nan,
            "r2": np.nan,
        }

    X = np.column_stack([np.ones(n), x_prev])
    coef, _, _, _ = np.linalg.lstsq(X, x_now, rcond=None)
    alpha, beta = coef
    fitted = X @ coef
    resid = x_now - fitted

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((x_now - np.mean(x_now)) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    shock_std = float(np.std(resid, ddof=1)) if n > 2 else np.nan

    if abs(1.0 - beta) > 1e-10:
        mu = float(alpha / (1.0 - beta))
    else:
        mu = np.nan

    if 0.0 < beta < 1.0:
        lambda_per_min = float(-math.log(beta))
        half_life_min = float(math.log(2.0) / lambda_per_min)
    else:
        lambda_per_min = np.nan
        half_life_min = np.nan

    return {
        "n_obs": n,
        "alpha": float(alpha),
        "beta": float(beta),
        "mu": mu,
        "lambda_per_min": lambda_per_min,
        "half_life_min": half_life_min,
        "shock_std": shock_std,
        "r2": r2,
    }



def fit_regime_models(df_trans: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        for regime in [0, 1]:
            g = df_trans[df_trans["regime_lag1"] == regime].copy()
            stats = fit_ar1_single_regime(g, feature)
            stats["feature"] = feature
            stats["regime"] = regime
            stats["regime_name"] = "fast" if regime == 1 else "slow"
            rows.append(stats)
    res = pd.DataFrame(rows)
    return res[[
        "feature", "regime", "regime_name", "n_obs", "alpha", "beta", "mu",
        "lambda_per_min", "half_life_min", "shock_std", "r2"
    ]].sort_values(["feature", "regime"]).reset_index(drop=True)



def compare_fast_slow(fit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, g in fit_df.groupby("feature"):
        slow = g[g["regime"] == 0].iloc[0]
        fast = g[g["regime"] == 1].iloc[0]
        rows.append({
            "feature": feature,
            "slow_lambda_per_min": slow["lambda_per_min"],
            "fast_lambda_per_min": fast["lambda_per_min"],
            "lambda_ratio_fast_to_slow": fast["lambda_per_min"] / slow["lambda_per_min"] if pd.notna(slow["lambda_per_min"]) and abs(slow["lambda_per_min"]) > 0 else np.nan,
            "slow_half_life_min": slow["half_life_min"],
            "fast_half_life_min": fast["half_life_min"],
            "slow_shock_std": slow["shock_std"],
            "fast_shock_std": fast["shock_std"],
            "shock_std_ratio_fast_to_slow": fast["shock_std"] / slow["shock_std"] if pd.notna(slow["shock_std"]) and abs(slow["shock_std"]) > 0 else np.nan,
            "slow_beta": slow["beta"],
            "fast_beta": fast["beta"],
            "slow_r2": slow["r2"],
            "fast_r2": fast["r2"],
        })
    return pd.DataFrame(rows)



def summarize_regimes(df: pd.DataFrame) -> pd.DataFrame:
    day = (
        df.groupby("trade_date", as_index=False)
        .agg(
            n_obs=("timestamp", "size"),
            n_fast=("regime", "sum"),
            n_triggers=("fast_trigger", "sum"),
            max_event_score=("event_score", "max"),
        )
    )
    day["pct_fast"] = 100.0 * day["n_fast"] / day["n_obs"]
    return day.sort_values(["max_event_score", "pct_fast"], ascending=False).reset_index(drop=True)



def choose_plot_day(df: pd.DataFrame, requested_day: str | None = None) -> pd.Timestamp:
    if requested_day is not None:
        day = pd.to_datetime(requested_day).normalize()
        return day
    tmp = (
        df.groupby("trade_date", as_index=False)
        .agg(max_event_score=("event_score", "max"), n_fast=("regime", "sum"))
        .sort_values(["max_event_score", "n_fast"], ascending=False)
        .reset_index(drop=True)
    )
    return tmp.loc[0, "trade_date"]



def plot_intraday_regimes(df: pd.DataFrame, day: pd.Timestamp, out_dir: Path) -> None:
    g = df[df["trade_date"] == day].copy().sort_values("timestamp")
    if g.empty:
        return

    def add_fast_shading(ax, gsub):
        in_fast = False
        start = None
        times = gsub["timestamp"].tolist()
        regs = gsub["regime"].tolist()
        for i, r in enumerate(regs):
            if (r == 1) and (not in_fast):
                in_fast = True
                start = times[i]
            last_point = (i == len(regs) - 1)
            if in_fast and ((r == 0) or last_point):
                end = times[i] if r == 0 else times[i]
                ax.axvspan(start, end, alpha=0.2)
                in_fast = False
                start = None

    for col, title, fname in [
        ("sigma_atm", f"ATM IV with fast-regime shading — {day.date()}", "regime_plot_sigma.png"),
        ("skew", f"Skew with fast-regime shading — {day.date()}", "regime_plot_skew.png"),
        ("curvature_bfly", f"Butterfly curvature with fast-regime shading — {day.date()}", "regime_plot_curvature.png"),
        ("event_score", f"Event score with fast-regime shading — {day.date()}", "regime_plot_score.png"),
    ]:
        plt.figure(figsize=(12, 4.2))
        ax = plt.gca()
        plt.plot(g["timestamp"], g[col], linewidth=1.4)
        trig = g[g["fast_trigger"] == 1]
        if not trig.empty:
            plt.scatter(trig["timestamp"], trig[col], s=22)
        add_fast_shading(ax, g)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=160, bbox_inches="tight")
        plt.close()


def main(
    INPUT_DIR: str = str(DEFAULT_INPUT_DIR),
    OUTPUT_DIR: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    ensure_dir(out_dir)

    feature_path = in_dir / FEATURES_FILE
    if not feature_path.exists():
        raise FileNotFoundError(f"Expected feature file: {feature_path}")

    df = load_features(feature_path)
    df = build_regime_labels(df)
    df_trans = add_lags(df, FEATURES)

    fit_df = fit_regime_models(df_trans, FEATURES)
    compare_df = compare_fast_slow(fit_df)
    regime_day_df = summarize_regimes(df)

    df.to_csv(out_dir / "regime_labeled_intraday_features_1dte.csv", index=False)
    df_trans.to_csv(out_dir / "regime_transition_panel_1dte.csv", index=False)
    fit_df.to_csv(out_dir / "regime_ar1_ou_fit_summary_1dte.csv", index=False)
    compare_df.to_csv(out_dir / "regime_fast_vs_slow_comparison_1dte.csv", index=False)
    regime_day_df.to_csv(out_dir / "regime_day_summary_1dte.csv", index=False)

    plot_day = choose_plot_day(df, PLOT_DAY)
    plot_intraday_regimes(df, plot_day, out_dir)

    print("Regime fit summary:")
    print(fit_df.to_string(index=False))
    print("\nFast vs slow comparison:")
    print(compare_df.to_string(index=False))
    print("\nTop regime days:")
    print(regime_day_df.head(10).to_string(index=False))
    print(f"\nPlotted day: {plot_day.date()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths / knobs
# ============================================================
QUOTES_FILE = "cleaned_quotes_with_iv_1dte.csv"
PRED_FEATURE_FILE = "boundary_layer_test_feature_predictions.csv"
DEFAULT_IV_INPUT_DIR = Path("IV_Out")
DEFAULT_FEATURE_INPUT_DIR = Path("Calibration_Out")
DEFAULT_OUTPUT_DIR = Path("Backtest_Out")

MIN_MID = 0.05
MAX_ABS_K_MAIN = 0.03
MAX_ABS_K_HEDGE = 0.02
VEGA_MIN_HEDGE = 0.25

# Regime-dependent execution
SLOW_REHEDGE_INTERVAL_MIN = 10
STOCK_BAND_DELTA = 0.05
STOCK_BAND_DELTAVEGA = 0.05
OPTION_BAND_DELTAVEGA = 0.05
FAST_REHEDGE_EVERY_MIN = True

# Optional costs
STOCK_TC_PER_SHARE = 0.00
OPTION_TC_PER_UNIT = 0.00

DELTA_BUMP_FRAC = 1e-4
SIGMA_BUMP = 0.005
MAX_ABS_HEDGE_OPTION_UNITS = 10.0


# ============================================================
# Pricing helpers
# ============================================================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(cp: str, S: float, K: float, tau: float, sigma: float, r: float = 0.0) -> float:
    S = max(float(S), 1e-12)
    K = max(float(K), 1e-12)
    tau = max(float(tau), 1e-10)
    sigma = min(max(float(sigma), 1e-8), 5.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)

    if str(cp).upper().startswith("C"):
        return S * norm_cdf(d1) - K * math.exp(-r * tau) * norm_cdf(d2)
    return K * math.exp(-r * tau) * norm_cdf(-d2) - S * norm_cdf(-d1)


def smile_iv(S: float, K: float, sigma_atm: float, skew: float, curvature: float,
             vol_floor: float = 1e-4, vol_cap: float = 3.0) -> float:
    k = math.log(max(float(K), 1e-12) / max(float(S), 1e-12))
    sigma = float(sigma_atm) + float(skew) * k + 0.5 * float(curvature) * k * k
    return min(max(sigma, vol_floor), vol_cap)


def surface_price(cp: str, S: float, K: float, tau: float,
                  sigma_atm: float, skew: float, curvature: float,
                  r: float = 0.0) -> float:
    sigma = smile_iv(S, K, sigma_atm, skew, curvature)
    return bs_price(cp, S, K, tau, sigma, r)


def surface_delta(cp: str, S: float, K: float, tau: float,
                  sigma_atm: float, skew: float, curvature: float,
                  r: float = 0.0, bump_frac: float = DELTA_BUMP_FRAC) -> float:
    h = max(abs(float(S)) * bump_frac, 1e-3)
    up = surface_price(cp, S + h, K, tau, sigma_atm, skew, curvature, r)
    dn = surface_price(cp, max(S - h, 1e-8), K, tau, sigma_atm, skew, curvature, r)
    return (up - dn) / (2.0 * h)


def surface_vega_atm(cp: str, S: float, K: float, tau: float,
                     sigma_atm: float, skew: float, curvature: float,
                     r: float = 0.0, bump: float = SIGMA_BUMP) -> float:
    up = surface_price(cp, S, K, tau, sigma_atm + bump, skew, curvature, r)
    dn = surface_price(cp, S, K, tau, sigma_atm - bump, skew, curvature, r)
    return (up - dn) / (2.0 * bump)


# ============================================================
# Data build
# ============================================================
def load_feature_panel(path: Path) -> pd.DataFrame:
    feat = pd.read_csv(path, parse_dates=["timestamp", "trade_date", "timestamp_next"])
    feat["trade_date"] = pd.to_datetime(feat["trade_date"]).dt.normalize()
    keep = [
        "timestamp", "trade_date", "regime", "regime_name", "event_score",
        "sigma_atm_pred", "skew_pred", "curvature_bfly_pred",
    ]
    return feat[keep].sort_values(["trade_date", "timestamp"]).reset_index(drop=True)


def load_quote_transitions(test_dates: set[pd.Timestamp], quotes_path: Path) -> pd.DataFrame:
    q = pd.read_csv(quotes_path, parse_dates=["timestamp", "trade_date", "expiry_date"])
    q["trade_date"] = pd.to_datetime(q["trade_date"]).dt.normalize()
    q["cp"] = q["cp"].astype(str)

    q = q[q["trade_date"].isin(test_dates)].copy()
    q = q.sort_values(["trade_date", "expiry_date", "cp", "K", "timestamp"]).reset_index(drop=True)

    same_contract_next = (
        q["trade_date"].eq(q["trade_date"].shift(-1))
        & q["expiry_date"].eq(q["expiry_date"].shift(-1))
        & q["cp"].eq(q["cp"].shift(-1))
        & q["K"].eq(q["K"].shift(-1))
    )
    one_min_next = (q["timestamp"].shift(-1) - q["timestamp"]).dt.total_seconds().eq(60)

    qt = q.loc[same_contract_next & one_min_next].copy()

    for col in ["timestamp", "mid", "S_used", "tau", "iv", "vega", "k"]:
        qt[f"{col}_next"] = q[col].shift(-1).loc[qt.index].values

    qt["contract_id"] = qt["cp"].astype(str) + "_" + qt["K"].round(6).astype(str)
    qt["abs_k"] = qt["k"].abs()

    qt = qt[(qt["mid"] > MIN_MID) & (qt["mid_next"] > MIN_MID) & (qt["abs_k"] <= MAX_ABS_K_MAIN)].copy()
    return qt.reset_index(drop=True)


def choose_fixed_hedges(qt: pd.DataFrame) -> pd.DataFrame:
    cand = qt[(qt["abs_k"] <= MAX_ABS_K_HEDGE) & (qt["vega"] >= VEGA_MIN_HEDGE)].copy()

    score = (
        cand.groupby(["trade_date", "expiry_date", "contract_id", "cp", "K"], as_index=False)
            .agg(
                coverage=("timestamp", "size"),
                mean_vega=("vega", "mean"),
                mean_abs_k=("abs_k", "mean"),
                mean_mid=("mid", "mean"),
            )
            .sort_values(
                ["trade_date", "expiry_date", "coverage", "mean_vega", "mean_abs_k"],
                ascending=[True, True, False, False, True],
            )
            .reset_index(drop=True)
    )
    score["rank"] = score.groupby(["trade_date", "expiry_date"]).cumcount() + 1
    return score[score["rank"].isin([1, 2])].copy()


def build_main_panel(qt: pd.DataFrame, feat: pd.DataFrame, hedge_choice: pd.DataFrame) -> pd.DataFrame:
    main = qt.merge(feat, on=["timestamp", "trade_date"], how="inner").copy()

    top1 = hedge_choice[hedge_choice["rank"] == 1][["trade_date", "expiry_date", "contract_id", "cp", "K"]].rename(
        columns={"contract_id": "hedge1_id", "cp": "hedge1_cp", "K": "hedge1_K"}
    )
    top2 = hedge_choice[hedge_choice["rank"] == 2][["trade_date", "expiry_date", "contract_id", "cp", "K"]].rename(
        columns={"contract_id": "hedge2_id", "cp": "hedge2_cp", "K": "hedge2_K"}
    )

    main = main.merge(top1, on=["trade_date", "expiry_date"], how="left")
    main = main.merge(top2, on=["trade_date", "expiry_date"], how="left")

    main["hedge_id"] = np.where(main["contract_id"] == main["hedge1_id"], main["hedge2_id"], main["hedge1_id"])

    hedge_lookup = qt[
        ["timestamp", "trade_date", "expiry_date", "contract_id", "cp", "K", "mid", "mid_next", "S_used", "S_used_next", "tau", "r", "iv", "vega", "k"]
    ].copy()
    hedge_lookup = hedge_lookup.rename(columns={c: f"{c}_h" for c in hedge_lookup.columns if c not in ["timestamp", "trade_date", "expiry_date"]})

    main = main.merge(
        hedge_lookup,
        left_on=["timestamp", "trade_date", "expiry_date", "hedge_id"],
        right_on=["timestamp", "trade_date", "expiry_date", "contract_id_h"],
        how="left",
    )

    return main[main["contract_id_h"].notna()].reset_index(drop=True)


# ============================================================
# Targets
# ============================================================
def compute_targets(main: pd.DataFrame) -> pd.DataFrame:
    n = len(main)
    delta_main = np.zeros(n)
    delta_hedge = np.zeros(n)
    vega_main = np.zeros(n)
    vega_hedge = np.zeros(n)

    for i, row in enumerate(main.itertuples(index=False)):
        a = float(row.sigma_atm_pred)
        b = float(row.skew_pred)
        c = float(row.curvature_bfly_pred)

        delta_main[i] = surface_delta(row.cp, row.S_used, row.K, row.tau, a, b, c, row.r)
        vega_main[i] = surface_vega_atm(row.cp, row.S_used, row.K, row.tau, a, b, c, row.r)

        delta_hedge[i] = surface_delta(row.cp_h, row.S_used_h, row.K_h, row.tau_h, a, b, c, row.r_h)
        vega_hedge[i] = surface_vega_atm(row.cp_h, row.S_used_h, row.K_h, row.tau_h, a, b, c, row.r_h)

    out = main.copy()
    out["target_stock_delta"] = -delta_main

    q_opt = -vega_main / np.where(np.abs(vega_hedge) > 1e-8, vega_hedge, np.nan)
    q_opt = np.clip(q_opt, -MAX_ABS_HEDGE_OPTION_UNITS, MAX_ABS_HEDGE_OPTION_UNITS)
    q_stock = -(delta_main + q_opt * delta_hedge)

    out["target_opt_deltavega"] = q_opt
    out["target_stock_deltavega"] = q_stock
    return out


# ============================================================
# Simulation
# ============================================================
def should_rehedge_regime(
    regime: int,
    timestamp: pd.Timestamp,
    last_rehedge_ts: pd.Timestamp | None,
    target_stock: float,
    held_stock: float,
    target_opt: float,
    held_opt: float,
    stock_band: float | None,
    opt_band: float | None,
) -> bool:
    if last_rehedge_ts is None:
        return True

    if FAST_REHEDGE_EVERY_MIN and int(regime) == 1:
        return True

    mins_since = (timestamp - last_rehedge_ts).total_seconds() / 60.0
    if mins_since >= SLOW_REHEDGE_INTERVAL_MIN:
        return True

    if stock_band is not None and stock_band > 0 and abs(target_stock - held_stock) >= stock_band:
        return True

    if opt_band is not None and opt_band > 0 and abs(target_opt - held_opt) >= opt_band:
        return True

    return False


def run_policy(main: pd.DataFrame, hedge_type: str, execution: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    total_rehedges = 0
    total_stock_turnover = 0.0
    total_option_turnover = 0.0

    for _, g in main.groupby(["trade_date", "expiry_date", "cp", "K"], sort=False):
        g = g.sort_values("timestamp")

        ts_arr = g["timestamp"].to_numpy()
        trade_date_arr = g["trade_date"].to_numpy()
        regime_arr = g["regime"].to_numpy(dtype=int)
        regime_name_arr = g["regime_name"].to_numpy()
        dS_arr = (g["S_used_next"] - g["S_used"]).to_numpy(dtype=float)
        dV_arr = (g["mid_next"] - g["mid"]).to_numpy(dtype=float)
        dH_arr = (g["mid_next_h"] - g["mid_h"]).to_numpy(dtype=float)

        if hedge_type == "delta":
            target_stock_arr = g["target_stock_delta"].to_numpy(dtype=float)
            target_opt_arr = np.zeros(len(g), dtype=float)
            stock_band = STOCK_BAND_DELTA
            opt_band = None
        else:
            target_stock_arr = g["target_stock_deltavega"].to_numpy(dtype=float)
            target_opt_arr = g["target_opt_deltavega"].to_numpy(dtype=float)
            stock_band = STOCK_BAND_DELTAVEGA
            opt_band = OPTION_BAND_DELTAVEGA

        held_stock = 0.0
        held_opt = 0.0
        last_rehedge_ts = None

        for i in range(len(g)):
            target_stock = float(target_stock_arr[i])
            target_opt = float(target_opt_arr[i])

            if execution == "every_min":
                if last_rehedge_ts is None:
                    do_rehedge = True
                else:
                    mins_since = (pd.Timestamp(ts_arr[i]) - last_rehedge_ts).total_seconds() / 60.0
                    do_rehedge = mins_since >= 10
                    
            else:
                do_rehedge = should_rehedge_regime(
                    regime=int(regime_arr[i]),
                    timestamp=pd.Timestamp(ts_arr[i]),
                    last_rehedge_ts=last_rehedge_ts,
                    target_stock=target_stock,
                    held_stock=held_stock,
                    target_opt=target_opt,
                    held_opt=held_opt,
                    stock_band=stock_band,
                    opt_band=opt_band,
                )

            d_stock_trade = 0.0
            d_opt_trade = 0.0

            if do_rehedge:
                d_stock_trade = target_stock - held_stock
                d_opt_trade = target_opt - held_opt

                total_stock_turnover += abs(d_stock_trade)
                total_option_turnover += abs(d_opt_trade)

                if abs(d_stock_trade) + abs(d_opt_trade) > 1e-12:
                    total_rehedges += 1

                held_stock = target_stock
                held_opt = target_opt
                last_rehedge_ts = pd.Timestamp(ts_arr[i])

            tc = STOCK_TC_PER_SHARE * abs(d_stock_trade) + OPTION_TC_PER_UNIT * abs(d_opt_trade)
            err = float(dV_arr[i] + held_stock * dS_arr[i] + held_opt * dH_arr[i] - tc)

            rows.append(
                {
                    "timestamp": ts_arr[i],
                    "trade_date": trade_date_arr[i],
                    "regime_name": regime_name_arr[i],
                    "method": f"{hedge_type}_{execution}",
                    "hedge_error": err,
                    "did_rehedge": bool(do_rehedge),
                    "trade_stock_abs": abs(d_stock_trade),
                    "trade_opt_abs": abs(d_opt_trade),
                }
            )

    panel = pd.DataFrame(rows)
    stats = {
        "method": f"{hedge_type}_{execution}",
        "n_obs": int(len(panel)),
        "mae": float(panel["hedge_error"].abs().mean()),
        "rmse": float(np.sqrt(np.mean(panel["hedge_error"] ** 2))),
        "p95_abs_err": float(panel["hedge_error"].abs().quantile(0.95)),
        "mean_err": float(panel["hedge_error"].mean()),
        "n_rehedges": int(total_rehedges),
        "stock_turnover": float(total_stock_turnover),
        "option_turnover": float(total_option_turnover),
        "avg_rehedges_per_day": float(total_rehedges / max(panel["trade_date"].nunique(), 1)),
    }
    return panel, stats


# ============================================================
# Summaries / plots
# ============================================================
def summarize_by_day(panel_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, trade_date), grp in panel_all.groupby(["method", "trade_date"]):
        rows.append(
            {
                "method": method,
                "trade_date": trade_date,
                "mae": float(grp["hedge_error"].abs().mean()),
                "rmse": float(np.sqrt(np.mean(grp["hedge_error"] ** 2))),
                "n_rehedges": int(grp["did_rehedge"].sum()),
                "stock_turnover": float(grp["trade_stock_abs"].sum()),
                "option_turnover": float(grp["trade_opt_abs"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_regime(panel_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, regime_name), grp in panel_all.groupby(["method", "regime_name"]):
        rows.append(
            {
                "method": method,
                "regime_name": regime_name,
                "n_obs": int(len(grp)),
                "mae": float(grp["hedge_error"].abs().mean()),
                "rmse": float(np.sqrt(np.mean(grp["hedge_error"] ** 2))),
                "p95_abs_err": float(grp["hedge_error"].abs().quantile(0.95)),
                "mean_err": float(grp["hedge_error"].mean()),
                "rehedge_rate": float(grp["did_rehedge"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_plots(panel_all: pd.DataFrame, daily: pd.DataFrame, overall: pd.DataFrame, out_dir: Path) -> None:
    cum = (
        panel_all.groupby(["timestamp", "method"], as_index=False)["hedge_error"]
                 .apply(lambda s: float(np.mean(np.abs(s))))
                 .rename(columns={"hedge_error": "mean_abs_err"})
    )
    cum = cum.sort_values(["method", "timestamp"])
    cum["cum_abs_err"] = cum.groupby("method")["mean_abs_err"].cumsum()

    plt.figure(figsize=(11, 4.5))
    for method, grp in cum.groupby("method"):
        plt.plot(grp["timestamp"], grp["cum_abs_err"], label=method)
    plt.title("Cumulative absolute hedge error")
    plt.xlabel("Timestamp")
    plt.ylabel("Cumulative mean absolute error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "regime_rehedge_v2_plot_cum_abs_error.png", dpi=160, bbox_inches="tight")
    plt.close()

    piv = daily.pivot(index="trade_date", columns="method", values="mae").sort_index()
    plt.figure(figsize=(11, 4.5))
    for c in piv.columns:
        plt.plot(piv.index, piv[c], marker="o", label=c)
    plt.title("Daily MAE by rehedging policy")
    plt.xlabel("Trade date")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "regime_rehedge_v2_plot_daily_mae.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10.5, 4.5))
    x = np.arange(len(overall))
    plt.bar(x - 0.18, overall["n_rehedges"].values, width=0.36, label="Rehedges")
    plt.bar(x + 0.18, overall["avg_rehedges_per_day"].values, width=0.36, label="Avg/day")
    plt.xticks(x, overall["method"].tolist(), rotation=20, ha="right")
    plt.title("Rehedge counts by policy")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "regime_rehedge_v2_plot_counts.png", dpi=160, bbox_inches="tight")
    plt.close()


def main(
    IV_INPUT_DIR: str = str(DEFAULT_IV_INPUT_DIR),
    FEATURE_INPUT_DIR: str = str(DEFAULT_FEATURE_INPUT_DIR),
    OUTPUT_DIR: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    iv_input_dir = Path(IV_INPUT_DIR)
    feat_input_dir = Path(FEATURE_INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)

    quotes_path = iv_input_dir / QUOTES_FILE
    pred_path = feat_input_dir / PRED_FEATURE_FILE

    if not quotes_path.exists():
        raise FileNotFoundError(f"Expected quote file: {quotes_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Expected prediction file: {pred_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    feat = load_feature_panel(pred_path)
    test_dates = set(pd.to_datetime(feat["trade_date"]).dt.normalize().unique())

    qt = load_quote_transitions(test_dates, quotes_path)
    hedge_choice = choose_fixed_hedges(qt)
    main = build_main_panel(qt, feat, hedge_choice)
    main = compute_targets(main)

    panels = []
    stats_rows = []

    for hedge_type, execution in [
        ("delta", "every_min"),
        ("delta", "regime"),
        ("deltavega", "every_min"),
        ("deltavega", "regime"),
    ]:
        panel_i, stats_i = run_policy(main, hedge_type=hedge_type, execution=execution)
        panels.append(panel_i)
        stats_rows.append(stats_i)

    panel_all = pd.concat(panels, axis=0, ignore_index=True)
    overall = pd.DataFrame(stats_rows).sort_values("method").reset_index(drop=True)
    by_day = summarize_by_day(panel_all)
    by_regime = summarize_by_regime(panel_all)

    reduction_rows = []
    for hedge_type in ["delta", "deltavega"]:
        base = overall.loc[overall["method"] == f"{hedge_type}_every_min"].iloc[0]
        rgm = overall.loc[overall["method"] == f"{hedge_type}_regime"].iloc[0]
        reduction_rows.append(
            {
                "hedge_type": hedge_type,
                "mae_ratio_regime_to_every": float(rgm["mae"] / base["mae"]),
                "rmse_ratio_regime_to_every": float(rgm["rmse"] / base["rmse"]),
                "rehedge_ratio_regime_to_every": float(rgm["n_rehedges"] / max(base["n_rehedges"], 1)),
                "stock_turnover_ratio_regime_to_every": float(rgm["stock_turnover"] / max(base["stock_turnover"], 1e-12)),
                "option_turnover_ratio_regime_to_every": float(rgm["option_turnover"] / max(base["option_turnover"], 1e-12)),
            }
        )
    reduction = pd.DataFrame(reduction_rows)

    hedge_choice.to_csv(out_dir / "regime_rehedge_v2_fixed_hedge_contracts.csv", index=False)
    overall.to_csv(out_dir / "regime_rehedge_v2_summary_overall.csv", index=False)
    by_day.to_csv(out_dir / "regime_rehedge_v2_summary_by_day.csv", index=False)
    by_regime.to_csv(out_dir / "regime_rehedge_v2_summary_by_regime.csv", index=False)
    reduction.to_csv(out_dir / "regime_rehedge_v2_reduction_vs_every_min.csv", index=False)

    make_plots(panel_all, by_day, overall, out_dir=out_dir)

    print("Overall summary:")
    print(overall.to_string(index=False))
    print("\nReduction vs every-minute:")
    print(reduction.to_string(index=False))


if __name__ == "__main__":
    main()

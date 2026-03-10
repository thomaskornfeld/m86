from __future__ import annotations
from tqdm.auto import tqdm

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Cost model inputs
# ============================================================
OPTION_CONTRACT_MULTIPLIER = 100.0

# Sourced explicit-fee defaults used previously:
# Nasdaq remove liquidity: $0.0030/share
# SEC Section 31: $20.60 per $1,000,000 of covered sales
# Central option assumption: $0.125/contract
SEC31_RATE_PER_DOLLAR_SELL = 20.60 / 1_000_000.0


def default_tc_config() -> dict:
    return {
        "stock_remove_fee_per_share": 0.0030,
        "stock_half_spread_per_share": 0.0,
        "sec31_rate_per_dollar_sell": SEC31_RATE_PER_DOLLAR_SELL,
        "option_fee_per_contract": 0.125,
        "option_half_spread_per_unit": 0.0,
        "option_contract_multiplier": OPTION_CONTRACT_MULTIPLIER,
    }


def low_tc_config() -> dict:
    cfg = default_tc_config()
    cfg["option_fee_per_contract"] = 0.065
    return cfg


def high_tc_config() -> dict:
    cfg = default_tc_config()
    cfg["option_fee_per_contract"] = 0.245
    return cfg


# ============================================================
# Paths / knobs
# ============================================================
QUOTES_CSV = Path("/Users/brendonbazzani/VS Code Projects-python/cleaned_quotes_with_iv_1dte.csv")
FEATURE_PRED_CSV = Path("/Users/brendonbazzani/VS Code Projects-python/boundary_layer_test_feature_predictions.csv")
OUT_DIR = Path("/Users/brendonbazzani/VS Code Projects-python")

MIN_MID = 0.05
MAX_ABS_K_MAIN = 0.03
MAX_ABS_K_HEDGE = 0.02
VEGA_MIN_HEDGE = 0.25

# Rehedge settings by hedge type
EVERY_MIN_INTERVALS_MIN = {
    "delta": 5,
    "deltavega": 5,
}

REGIME_FAST_EVERY_MIN = {
    "delta": True,
    "deltavega": True,
}

REGIME_SLOW_INTERVAL_MIN = {
    "delta": 10,
    "deltavega": 10,
}

STOCK_BANDS = {
    "delta": 0.05,
    "deltavega": 0.05,
}

OPTION_BANDS = {
    "delta": None,
    "deltavega": 0.05,
}

DELTA_BUMP_FRAC = 1e-4
SIGMA_BUMP = 0.005
MAX_ABS_HEDGE_OPTION_UNITS = 10.0

# ============================================================
# Tuning controls
# ============================================================
USE_TUNING = False
SAVE_TUNED_CONFIGS = False

# If USE_TUNING = False, these are the fixed configs used.
MANUAL_REGIME_CONFIGS = {
    "delta": {
        "hedge_type": "delta",
        "every_interval_min": 1.0,
        "slow_interval_min": 20.0,
        "stock_band": 0.1,
        "option_band": None,
        "fast_every_min": False,
    },
    "deltavega": {
        "hedge_type": "deltavega",
        "every_interval_min": 1.0,
        "slow_interval_min": 20.0,
        "stock_band": 0.1,
        "option_band": 0.1,
        "fast_every_min": False,
    },
}

# Tuning grids
DELTA_TUNING_GRID = {
    "slow_interval_grid": [5, 10, 20],
    "stock_band_grid": [0.02, 0.05, 0.10],
    "option_band_grid": [None],
    "fast_every_min_grid": [True, False],
}

DELTAVEGA_TUNING_GRID = {
    "slow_interval_grid": [5, 10, 20],
    "stock_band_grid": [0.02, 0.05, 0.10],
    "option_band_grid": [0.02, 0.05, 0.10],
    "fast_every_min_grid": [True, False],
}

ALLOWED_MAE_INCREASE = {
    "delta": 0.05,
    "deltavega": 0.05,
}

# ============================================================
# Regime config helpers
# ============================================================
def build_regime_config(
    hedge_type: str,
    every_interval_min: float = 1.0,
    slow_interval_min: float = 10.0,
    stock_band: float | None = 0.05,
    option_band: float | None = None,
    fast_every_min: bool = True,
) -> dict:
    return {
        "hedge_type": hedge_type,
        "every_interval_min": float(every_interval_min),
        "slow_interval_min": float(slow_interval_min),
        "stock_band": None if stock_band is None else float(stock_band),
        "option_band": None if option_band is None else float(option_band),
        "fast_every_min": bool(fast_every_min),
    }


def default_regime_config_from_globals(hedge_type: str) -> dict:
    return build_regime_config(
        hedge_type=hedge_type,
        every_interval_min=float(EVERY_MIN_INTERVALS_MIN[hedge_type]),
        slow_interval_min=float(REGIME_SLOW_INTERVAL_MIN[hedge_type]),
        stock_band=STOCK_BANDS[hedge_type],
        option_band=OPTION_BANDS[hedge_type],
        fast_every_min=bool(REGIME_FAST_EVERY_MIN[hedge_type]),
    )


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


def smile_iv(
    S: float,
    K: float,
    sigma_atm: float,
    skew: float,
    curvature: float,
    vol_floor: float = 1e-4,
    vol_cap: float = 3.0,
) -> float:
    k = math.log(max(float(K), 1e-12) / max(float(S), 1e-12))
    sigma = float(sigma_atm) + float(skew) * k + 0.5 * float(curvature) * k * k
    return min(max(sigma, vol_floor), vol_cap)


def surface_price(
    cp: str,
    S: float,
    K: float,
    tau: float,
    sigma_atm: float,
    skew: float,
    curvature: float,
    r: float = 0.0,
) -> float:
    sigma = smile_iv(S, K, sigma_atm, skew, curvature)
    return bs_price(cp, S, K, tau, sigma, r)


def surface_delta(
    cp: str,
    S: float,
    K: float,
    tau: float,
    sigma_atm: float,
    skew: float,
    curvature: float,
    r: float = 0.0,
    bump_frac: float = DELTA_BUMP_FRAC,
) -> float:
    h = max(abs(float(S)) * bump_frac, 1e-3)
    up = surface_price(cp, S + h, K, tau, sigma_atm, skew, curvature, r)
    dn = surface_price(cp, max(S - h, 1e-8), K, tau, sigma_atm, skew, curvature, r)
    return (up - dn) / (2.0 * h)


def surface_vega_atm(
    cp: str,
    S: float,
    K: float,
    tau: float,
    sigma_atm: float,
    skew: float,
    curvature: float,
    r: float = 0.0,
    bump: float = SIGMA_BUMP,
) -> float:
    up = surface_price(cp, S, K, tau, sigma_atm + bump, skew, curvature, r)
    dn = surface_price(cp, S, K, tau, sigma_atm - bump, skew, curvature, r)
    return (up - dn) / (2.0 * bump)


# ============================================================
# Data build
# ============================================================
def load_feature_panel() -> pd.DataFrame:
    feat = pd.read_csv(FEATURE_PRED_CSV, parse_dates=["timestamp", "trade_date", "timestamp_next"])
    feat["trade_date"] = pd.to_datetime(feat["trade_date"]).dt.normalize()
    keep = [
        "timestamp",
        "trade_date",
        "regime",
        "regime_name",
        "event_score",
        "sigma_atm_pred",
        "skew_pred",
        "curvature_bfly_pred",
    ]
    return feat[keep].sort_values(["trade_date", "timestamp"]).reset_index(drop=True)


def load_quote_transitions(test_dates: set[pd.Timestamp]) -> pd.DataFrame:
    q = pd.read_csv(QUOTES_CSV, parse_dates=["timestamp", "trade_date", "expiry_date"])
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

    qt = qt[
        (qt["mid"] > MIN_MID)
        & (qt["mid_next"] > MIN_MID)
        & (qt["abs_k"] <= MAX_ABS_K_MAIN)
    ].copy()

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

    top1 = hedge_choice[hedge_choice["rank"] == 1][
        ["trade_date", "expiry_date", "contract_id", "cp", "K"]
    ].rename(columns={"contract_id": "hedge1_id", "cp": "hedge1_cp", "K": "hedge1_K"})

    top2 = hedge_choice[hedge_choice["rank"] == 2][
        ["trade_date", "expiry_date", "contract_id", "cp", "K"]
    ].rename(columns={"contract_id": "hedge2_id", "cp": "hedge2_cp", "K": "hedge2_K"})

    main = main.merge(top1, on=["trade_date", "expiry_date"], how="left")
    main = main.merge(top2, on=["trade_date", "expiry_date"], how="left")

    main["hedge_id"] = np.where(
        main["contract_id"] == main["hedge1_id"],
        main["hedge2_id"],
        main["hedge1_id"],
    )

    hedge_lookup = qt[
        [
            "timestamp",
            "trade_date",
            "expiry_date",
            "contract_id",
            "cp",
            "K",
            "mid",
            "mid_next",
            "S_used",
            "S_used_next",
            "tau",
            "r",
            "iv",
            "vega",
            "k",
        ]
    ].copy()

    hedge_lookup = hedge_lookup.rename(
        columns={c: f"{c}_h" for c in hedge_lookup.columns if c not in ["timestamp", "trade_date", "expiry_date"]}
    )

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
# Trade-cost helpers
# ============================================================
def stock_trade_cost(d_stock_trade: float, stock_price: float, tc_cfg: dict) -> float:
    shares = abs(float(d_stock_trade))
    sell_shares = max(-float(d_stock_trade), 0.0)

    remove_fee = shares * float(tc_cfg["stock_remove_fee_per_share"])
    half_spread = shares * float(tc_cfg["stock_half_spread_per_share"])
    sec31_fee = sell_shares * float(stock_price) * float(tc_cfg["sec31_rate_per_dollar_sell"])

    return remove_fee + half_spread + sec31_fee


def option_trade_cost(d_opt_trade: float, tc_cfg: dict) -> float:
    units = abs(float(d_opt_trade))
    per_unit_explicit = float(tc_cfg["option_fee_per_contract"]) / float(tc_cfg["option_contract_multiplier"])
    half_spread = float(tc_cfg["option_half_spread_per_unit"])
    return units * (per_unit_explicit + half_spread)


def total_trade_cost(
    d_stock_trade: float,
    d_opt_trade: float,
    stock_price: float,
    tc_cfg: dict,
) -> float:
    return stock_trade_cost(d_stock_trade, stock_price, tc_cfg) + option_trade_cost(d_opt_trade, tc_cfg)


# ============================================================
# Simulation helpers
# ============================================================
def should_rehedge_regime_cfg(
    regime: int,
    timestamp: pd.Timestamp,
    last_rehedge_ts: pd.Timestamp | None,
    target_stock: float,
    held_stock: float,
    target_opt: float,
    held_opt: float,
    regime_cfg: dict,
) -> bool:
    if last_rehedge_ts is None:
        return True

    if bool(regime_cfg["fast_every_min"]) and int(regime) == 1:
        return True

    mins_since = (timestamp - last_rehedge_ts).total_seconds() / 60.0
    if mins_since >= float(regime_cfg["slow_interval_min"]):
        return True

    stock_band = regime_cfg["stock_band"]
    opt_band = regime_cfg["option_band"]

    if stock_band is not None and stock_band > 0 and abs(target_stock - held_stock) >= stock_band:
        return True

    if opt_band is not None and opt_band > 0 and abs(target_opt - held_opt) >= opt_band:
        return True

    return False


# ============================================================
# Simulation
# ============================================================
def run_policy(
    main: pd.DataFrame,
    hedge_type: str,
    execution: str,
    tc_cfg: dict | None = None,
    regime_cfg: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    if tc_cfg is None:
        tc_cfg = default_tc_config()
    if regime_cfg is None:
        regime_cfg = default_regime_config_from_globals(hedge_type)

    rows = []
    total_rehedges = 0
    total_stock_turnover = 0.0
    total_option_turnover = 0.0
    total_tc = 0.0

    for _, g in main.groupby(["trade_date", "expiry_date", "cp", "K"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        path_id = (
            g["trade_date"].iloc[0].strftime("%Y-%m-%d")
            + "_"
            + pd.Timestamp(g["expiry_date"].iloc[0]).strftime("%Y-%m-%d")
            + "_"
            + str(g["cp"].iloc[0])
            + "_"
            + f"{float(g['K'].iloc[0]):.6f}"
        )

        ts_arr = g["timestamp"].to_numpy()
        trade_date_arr = g["trade_date"].to_numpy()
        regime_arr = g["regime"].to_numpy(dtype=int)
        regime_name_arr = g["regime_name"].to_numpy()

        S_arr = g["S_used"].to_numpy(dtype=float)
        dS_arr = (g["S_used_next"] - g["S_used"]).to_numpy(dtype=float)
        dV_arr = (g["mid_next"] - g["mid"]).to_numpy(dtype=float)
        dH_arr = (g["mid_next_h"] - g["mid_h"]).to_numpy(dtype=float)

        if hedge_type == "delta":
            target_stock_arr = g["target_stock_delta"].to_numpy(dtype=float)
            target_opt_arr = np.zeros(len(g), dtype=float)
        else:
            target_stock_arr = g["target_stock_deltavega"].to_numpy(dtype=float)
            target_opt_arr = g["target_opt_deltavega"].to_numpy(dtype=float)

        held_stock = 0.0
        held_opt = 0.0
        last_rehedge_ts = None

        for i in range(len(g)):
            timestamp_i = pd.Timestamp(ts_arr[i])
            target_stock = float(target_stock_arr[i])
            target_opt = float(target_opt_arr[i])

            if execution == "every_min":
                if last_rehedge_ts is None:
                    do_rehedge = True
                else:
                    mins_since = (timestamp_i - last_rehedge_ts).total_seconds() / 60.0
                    do_rehedge = mins_since >= float(regime_cfg["every_interval_min"])
            else:
                do_rehedge = should_rehedge_regime_cfg(
                    regime=int(regime_arr[i]),
                    timestamp=timestamp_i,
                    last_rehedge_ts=last_rehedge_ts,
                    target_stock=target_stock,
                    held_stock=held_stock,
                    target_opt=target_opt,
                    held_opt=held_opt,
                    regime_cfg=regime_cfg,
                )

            d_stock_trade = 0.0
            d_opt_trade = 0.0
            tc = 0.0

            if do_rehedge:
                d_stock_trade = target_stock - held_stock
                d_opt_trade = target_opt - held_opt

                total_stock_turnover += abs(d_stock_trade)
                total_option_turnover += abs(d_opt_trade)

                tc = total_trade_cost(
                    d_stock_trade=d_stock_trade,
                    d_opt_trade=d_opt_trade,
                    stock_price=float(S_arr[i]),
                    tc_cfg=tc_cfg,
                )
                total_tc += tc

                if abs(d_stock_trade) + abs(d_opt_trade) > 1e-12:
                    total_rehedges += 1

                held_stock = target_stock
                held_opt = target_opt
                last_rehedge_ts = timestamp_i

            err = float(dV_arr[i] + held_stock * dS_arr[i] + held_opt * dH_arr[i] - tc)

            rows.append(
                {
                    "timestamp": ts_arr[i],
                    "trade_date": trade_date_arr[i],
                    "regime_name": regime_name_arr[i],
                    "method": f"{hedge_type}_{execution}",
                    "path_id": path_id,
                    "hedge_error": err,
                    "did_rehedge": bool(do_rehedge),
                    "trade_stock_abs": abs(d_stock_trade),
                    "trade_opt_abs": abs(d_opt_trade),
                    "trade_tc": float(tc),
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
        "total_tc": float(total_tc),
        "avg_tc_per_day": float(total_tc / max(panel["trade_date"].nunique(), 1)),
        "avg_rehedges_per_day": float(total_rehedges / max(panel["trade_date"].nunique(), 1)),
    }
    return panel, stats


# ============================================================
# Date split
# ============================================================
def split_main_by_date(
    main: pd.DataFrame,
    train_frac: float = 0.60,
    valid_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(main["trade_date"]).dt.normalize().unique())
    n = len(dates)
    if n < 3:
        raise ValueError("Need at least 3 trade dates for train/validation/test split.")

    n_train = max(1, int(math.floor(n * train_frac)))
    n_valid = max(1, int(math.floor(n * valid_frac)))

    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    if n_train + n_valid >= n:
        n_train = max(1, n - n_valid - 1)

    train_dates = set(dates[:n_train])
    valid_dates = set(dates[n_train:n_train + n_valid])
    test_dates = set(dates[n_train + n_valid:])

    train_main = main[main["trade_date"].isin(train_dates)].copy()
    valid_main = main[main["trade_date"].isin(valid_dates)].copy()
    test_main = main[main["trade_date"].isin(test_dates)].copy()

    return train_main, valid_main, test_main


# ============================================================
# Threshold construction from every-minute baseline
# ============================================================
def make_relative_mae_threshold(
    main: pd.DataFrame,
    hedge_type: str,
    tc_cfg: dict | None = None,
    allowed_increase: float = 0.05,
    train_frac: float = 0.60,
    valid_frac: float = 0.20,
) -> tuple[float, dict]:
    _, valid_main, _ = split_main_by_date(main, train_frac=train_frac, valid_frac=valid_frac)

    regime_cfg = default_regime_config_from_globals(hedge_type)
    regime_cfg["every_interval_min"] = 1.0

    _, base_stats = run_policy(
        valid_main,
        hedge_type=hedge_type,
        execution="every_min",
        tc_cfg=tc_cfg,
        regime_cfg=regime_cfg,
    )
    threshold = float(base_stats["mae"]) * (1.0 + float(allowed_increase))
    return threshold, base_stats


# ============================================================
# Grid search
# ============================================================
def tune_regime_to_mae_threshold(
    main: pd.DataFrame,
    hedge_type: str,
    mae_threshold: float,
    tc_cfg: dict | None,
    slow_interval_grid: Iterable[float],
    stock_band_grid: Iterable[float],
    option_band_grid: Iterable[float | None] | None = None,
    fast_every_min_grid: Iterable[bool] = (True,),
    train_frac: float = 0.60,
    valid_frac: float = 0.20,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    _, valid_main, test_main = split_main_by_date(
        main,
        train_frac=train_frac,
        valid_frac=valid_frac,
    )

    slow_interval_grid = list(slow_interval_grid)
    stock_band_grid = list(stock_band_grid)
    fast_every_min_grid = list(fast_every_min_grid)

    if option_band_grid is None:
        option_band_grid = [None]
    else:
        option_band_grid = list(option_band_grid)

    jobs = []
    for slow_interval_min in slow_interval_grid:
        for stock_band in stock_band_grid:
            for option_band in option_band_grid:
                for fast_every_min in fast_every_min_grid:
                    jobs.append((slow_interval_min, stock_band, option_band, fast_every_min))

    rows = []

    for slow_interval_min, stock_band, option_band, fast_every_min in tqdm(
        jobs,
        desc=f"Tuning {hedge_type}",
        total=len(jobs),
    ):
        regime_cfg = build_regime_config(
            hedge_type=hedge_type,
            every_interval_min=1.0,
            slow_interval_min=slow_interval_min,
            stock_band=stock_band,
            option_band=option_band,
            fast_every_min=fast_every_min,
        )

        _, stats_valid = run_policy(
            valid_main,
            hedge_type=hedge_type,
            execution="regime",
            tc_cfg=tc_cfg,
            regime_cfg=regime_cfg,
        )

        feasible = bool(stats_valid["mae"] <= mae_threshold)

        rows.append(
            {
                "hedge_type": hedge_type,
                "slow_interval_min": float(slow_interval_min),
                "stock_band": float(stock_band),
                "option_band": np.nan if option_band is None else float(option_band),
                "fast_every_min": bool(fast_every_min),
                "valid_feasible": feasible,
                "valid_mae": float(stats_valid["mae"]),
                "valid_rmse": float(stats_valid["rmse"]),
                "valid_p95_abs_err": float(stats_valid["p95_abs_err"]),
                "valid_n_rehedges": int(stats_valid["n_rehedges"]),
                "valid_stock_turnover": float(stats_valid["stock_turnover"]),
                "valid_option_turnover": float(stats_valid["option_turnover"]),
                "valid_total_tc": float(stats_valid["total_tc"]),
                "valid_avg_tc_per_day": float(stats_valid["avg_tc_per_day"]),
            }
        )

    results = pd.DataFrame(rows)

    feasible = results[results["valid_feasible"]].copy()
    if len(feasible) > 0:
        feasible = feasible.sort_values(
            [
                "valid_total_tc",
                "valid_n_rehedges",
                "valid_stock_turnover",
                "valid_option_turnover",
                "valid_mae",
            ],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
        winner = feasible.iloc[0]
    else:
        results = results.sort_values(
            [
                "valid_mae",
                "valid_total_tc",
                "valid_n_rehedges",
            ],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        winner = results.iloc[0]

    best_cfg = build_regime_config(
        hedge_type=hedge_type,
        every_interval_min=1.0,
        slow_interval_min=float(winner["slow_interval_min"]),
        stock_band=float(winner["stock_band"]),
        option_band=None if pd.isna(winner["option_band"]) else float(winner["option_band"]),
        fast_every_min=bool(winner["fast_every_min"]),
    )

    _, best_valid_stats = run_policy(
        valid_main,
        hedge_type=hedge_type,
        execution="regime",
        tc_cfg=tc_cfg,
        regime_cfg=best_cfg,
    )
    _, best_test_stats = run_policy(
        test_main,
        hedge_type=hedge_type,
        execution="regime",
        tc_cfg=tc_cfg,
        regime_cfg=best_cfg,
    )

    return results, best_cfg, best_valid_stats, best_test_stats

def tune_or_load_regime_configs(main_panel: pd.DataFrame, tc_cfg: dict) -> tuple[dict, pd.DataFrame]:
    chosen_cfgs: dict[str, dict] = {}
    summary_rows = []

    if USE_TUNING:
        for hedge_type, grid in [
            ("delta", DELTA_TUNING_GRID),
            ("deltavega", DELTAVEGA_TUNING_GRID),
        ]:
            mae_threshold, baseline_valid = make_relative_mae_threshold(
                main_panel,
                hedge_type=hedge_type,
                tc_cfg=tc_cfg,
                allowed_increase=ALLOWED_MAE_INCREASE[hedge_type],
            )

            results, best_cfg, best_valid_stats, best_test_stats = tune_regime_to_mae_threshold(
                main=main_panel,
                hedge_type=hedge_type,
                mae_threshold=mae_threshold,
                tc_cfg=tc_cfg,
                slow_interval_grid=grid["slow_interval_grid"],
                stock_band_grid=grid["stock_band_grid"],
                option_band_grid=grid["option_band_grid"],
                fast_every_min_grid=grid["fast_every_min_grid"],
            )

            results.to_csv(OUT_DIR / f"tc_threshold_tuning_{hedge_type}.csv", index=False)
            chosen_cfgs[hedge_type] = best_cfg

            summary_rows.append(
                {
                    "hedge_type": hedge_type,
                    "source": "tuned",
                    "mae_threshold": float(mae_threshold),
                    "baseline_valid_mae": float(baseline_valid["mae"]),
                    "slow_interval_min": float(best_cfg["slow_interval_min"]),
                    "stock_band": float(best_cfg["stock_band"]),
                    "option_band": np.nan if best_cfg["option_band"] is None else float(best_cfg["option_band"]),
                    "fast_every_min": bool(best_cfg["fast_every_min"]),
                    "valid_mae": float(best_valid_stats["mae"]),
                    "valid_rmse": float(best_valid_stats["rmse"]),
                    "valid_n_rehedges": int(best_valid_stats["n_rehedges"]),
                    "valid_total_tc": float(best_valid_stats["total_tc"]),
                    "test_mae": float(best_test_stats["mae"]),
                    "test_rmse": float(best_test_stats["rmse"]),
                    "test_n_rehedges": int(best_test_stats["n_rehedges"]),
                    "test_total_tc": float(best_test_stats["total_tc"]),
                }
            )

            print(f"\nChosen tuned config for {hedge_type}:")
            print(best_cfg)
            print("Validation stats:")
            print(best_valid_stats)
            print("Test stats:")
            print(best_test_stats)

    else:
        for hedge_type in ["delta", "deltavega"]:
            best_cfg = MANUAL_REGIME_CONFIGS[hedge_type]
            chosen_cfgs[hedge_type] = best_cfg

            _, valid_main, test_main = split_main_by_date(main_panel)

            _, valid_stats = run_policy(
                valid_main,
                hedge_type=hedge_type,
                execution="regime",
                tc_cfg=tc_cfg,
                regime_cfg=best_cfg,
            )
            _, test_stats = run_policy(
                test_main,
                hedge_type=hedge_type,
                execution="regime",
                tc_cfg=tc_cfg,
                regime_cfg=best_cfg,
            )

            summary_rows.append(
                {
                    "hedge_type": hedge_type,
                    "source": "manual",
                    "mae_threshold": np.nan,
                    "baseline_valid_mae": np.nan,
                    "slow_interval_min": float(best_cfg["slow_interval_min"]),
                    "stock_band": float(best_cfg["stock_band"]),
                    "option_band": np.nan if best_cfg["option_band"] is None else float(best_cfg["option_band"]),
                    "fast_every_min": bool(best_cfg["fast_every_min"]),
                    "valid_mae": float(valid_stats["mae"]),
                    "valid_rmse": float(valid_stats["rmse"]),
                    "valid_n_rehedges": int(valid_stats["n_rehedges"]),
                    "valid_total_tc": float(valid_stats["total_tc"]),
                    "test_mae": float(test_stats["mae"]),
                    "test_rmse": float(test_stats["rmse"]),
                    "test_n_rehedges": int(test_stats["n_rehedges"]),
                    "test_total_tc": float(test_stats["total_tc"]),
                }
            )

            print(f"\nUsing manual config for {hedge_type}:")
            print(best_cfg)
            print("Validation stats:")
            print(valid_stats)
            print("Test stats:")
            print(test_stats)

    chosen_df = pd.DataFrame(summary_rows)
    if SAVE_TUNED_CONFIGS:
        chosen_df.to_csv(OUT_DIR / "chosen_regime_configs.csv", index=False)

    return chosen_cfgs, chosen_df

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
                "total_tc": float(grp["trade_tc"].sum()) if "trade_tc" in grp.columns else np.nan,
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
                "total_tc": float(grp["trade_tc"].sum()) if "trade_tc" in grp.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_rehedge_timing(panel_all: pd.DataFrame) -> pd.DataFrame:
    tmp = panel_all.loc[panel_all["did_rehedge"], ["method", "path_id", "timestamp"]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    tmp = tmp.sort_values(["method", "path_id", "timestamp"])

    tmp["mins_since_prev_rehedge"] = (
        tmp.groupby(["method", "path_id"])["timestamp"].diff().dt.total_seconds() / 60.0
    )

    out = (
        tmp.dropna(subset=["mins_since_prev_rehedge"])
        .groupby("method", as_index=False)["mins_since_prev_rehedge"]
        .agg(
            mean_rehedge_mins="mean",
            median_rehedge_mins="median",
            p90_rehedge_mins=lambda s: s.quantile(0.90),
            max_rehedge_mins="max",
        )
    )
    return out


def make_plots(panel_all: pd.DataFrame, daily: pd.DataFrame, overall: pd.DataFrame) -> None:
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
    plt.savefig(OUT_DIR / "regime_rehedge_v2_plot_cum_abs_error.png", dpi=160, bbox_inches="tight")
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
    plt.savefig(OUT_DIR / "regime_rehedge_v2_plot_daily_mae.png", dpi=160, bbox_inches="tight")
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
    plt.savefig(OUT_DIR / "regime_rehedge_v2_plot_counts.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_rehedge_timing(timing: pd.DataFrame) -> None:
    if len(timing) == 0:
        return

    plt.figure(figsize=(10.5, 4.8))
    x = np.arange(len(timing))
    plt.bar(x - 0.18, timing["mean_rehedge_mins"], width=0.36, label="Mean minutes")
    plt.bar(x + 0.18, timing["median_rehedge_mins"], width=0.36, label="Median minutes")
    plt.xticks(x, timing["method"].tolist(), rotation=20, ha="right")
    plt.ylabel("Minutes")
    plt.title("Observed time between rehedges")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regime_rehedge_v2_plot_rehedge_timing.png", dpi=160, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main(
        QUOTES_CSV : str = Path("/cleaned_quotes_with_iv_1dte.csv"),
        FEATURE_PRED_CSV : str = Path("/boundary_layer_test_feature_predictions.csv"),
        OUT_DIR = Path("")
        ) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feat = load_feature_panel()
    test_dates = set(pd.to_datetime(feat["trade_date"]).dt.normalize().unique())

    qt = load_quote_transitions(test_dates)
    hedge_choice = choose_fixed_hedges(qt)
    main_panel = build_main_panel(qt, feat, hedge_choice)
    main_panel = compute_targets(main_panel)

    tc_cfg = default_tc_config()

    # ------------------------------------------------
    # Tune or load both regime configs
    # ------------------------------------------------
    chosen_cfgs, chosen_df = tune_or_load_regime_configs(main_panel, tc_cfg)

    print("\nFinal chosen regime levels:")
    print(chosen_df.to_string(index=False))

    # ------------------------------------------------
    # Run policies using chosen configs
    # ------------------------------------------------
    panels = []
    stats_rows = []

    for hedge_type, execution in [
        ("delta", "every_min"),
        ("delta", "regime"),
        ("deltavega", "every_min"),
        ("deltavega", "regime"),
    ]:
        if execution == "regime":
            panel_i, stats_i = run_policy(
                main_panel,
                hedge_type=hedge_type,
                execution=execution,
                tc_cfg=tc_cfg,
                regime_cfg=chosen_cfgs[hedge_type],
            )
        else:
            panel_i, stats_i = run_policy(
                main_panel,
                hedge_type=hedge_type,
                execution=execution,
                tc_cfg=tc_cfg,
            )

        panels.append(panel_i)
        stats_rows.append(stats_i)

    panel_all = pd.concat(panels, axis=0, ignore_index=True)
    overall = pd.DataFrame(stats_rows).sort_values("method").reset_index(drop=True)
    by_day = summarize_by_day(panel_all)
    by_regime = summarize_by_regime(panel_all)
    timing = summarize_rehedge_timing(panel_all)

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
                "tc_ratio_regime_to_every": float(rgm["total_tc"] / max(base["total_tc"], 1e-12)),
            }
        )
    reduction = pd.DataFrame(reduction_rows)

    hedge_choice.to_csv(OUT_DIR / "regime_rehedge_v2_fixed_hedge_contracts.csv", index=False)
    overall.to_csv(OUT_DIR / "regime_rehedge_v2_summary_overall.csv", index=False)
    by_day.to_csv(OUT_DIR / "regime_rehedge_v2_summary_by_day.csv", index=False)
    by_regime.to_csv(OUT_DIR / "regime_rehedge_v2_summary_by_regime.csv", index=False)
    reduction.to_csv(OUT_DIR / "regime_rehedge_v2_reduction_vs_every_min.csv", index=False)
    timing.to_csv(OUT_DIR / "regime_rehedge_v2_summary_rehedge_timing.csv", index=False)

    make_plots(panel_all, by_day, overall)
    plot_rehedge_timing(timing)

    compare_rows = []
    for hedge_type in ["delta", "deltavega"]:
        base = overall.loc[overall["method"] == f"{hedge_type}_every_min"].iloc[0]
        rgm = overall.loc[overall["method"] == f"{hedge_type}_regime"].iloc[0]

        compare_rows.append(
            {
                "hedge_type": hedge_type,
                "baseline_method": f"{hedge_type}_every_min",
                "regime_method": f"{hedge_type}_regime",
                "baseline_mae": float(base["mae"]),
                "regime_mae": float(rgm["mae"]),
                "mae_pct_change": 100.0 * (float(rgm["mae"]) / float(base["mae"]) - 1.0),
                "baseline_rmse": float(base["rmse"]),
                "regime_rmse": float(rgm["rmse"]),
                "rmse_pct_change": 100.0 * (float(rgm["rmse"]) / float(base["rmse"]) - 1.0),
                "baseline_p95_abs_err": float(base["p95_abs_err"]),
                "regime_p95_abs_err": float(rgm["p95_abs_err"]),
                "p95_pct_change": 100.0 * (float(rgm["p95_abs_err"]) / float(base["p95_abs_err"]) - 1.0),
                "baseline_rehedges": int(base["n_rehedges"]),
                "regime_rehedges": int(rgm["n_rehedges"]),
                "rehedges_pct_change": 100.0 * (float(rgm["n_rehedges"]) / max(float(base["n_rehedges"]), 1.0) - 1.0),
                "baseline_stock_turnover": float(base["stock_turnover"]),
                "regime_stock_turnover": float(rgm["stock_turnover"]),
                "stock_turnover_pct_change": 100.0 * (float(rgm["stock_turnover"]) / max(float(base["stock_turnover"]), 1e-12) - 1.0),
                "baseline_option_turnover": float(base["option_turnover"]),
                "regime_option_turnover": float(rgm["option_turnover"]),
                "option_turnover_pct_change": 100.0 * (float(rgm["option_turnover"]) / max(float(base["option_turnover"]), 1e-12) - 1.0),
                "baseline_total_tc": float(base["total_tc"]),
                "regime_total_tc": float(rgm["total_tc"]),
                "tc_pct_change": 100.0 * (float(rgm["total_tc"]) / max(float(base["total_tc"]), 1e-12) - 1.0),
            }
        )

    compare = pd.DataFrame(compare_rows)
    compare.to_csv(OUT_DIR / "regime_rehedge_v2_compare_pairs.csv", index=False)

    print("\nPairwise comparison:")
    print(compare.to_string(index=False))

    plt.figure(figsize=(9.5, 5.5))
    plt.scatter(overall["n_rehedges"], overall["mae"], s=120)
    for _, row in overall.iterrows():
        plt.annotate(row["method"], (row["n_rehedges"], row["mae"]), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("Number of rehedges")
    plt.ylabel("MAE of hedge error")
    plt.title("Tradeoff: hedge error vs rehedge count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regime_rehedge_v2_plot_tradeoff_mae_vs_rehedges.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9.5, 5.5))
    plt.scatter(overall["n_rehedges"], overall["rmse"], s=120)
    for _, row in overall.iterrows():
        plt.annotate(row["method"], (row["n_rehedges"], row["rmse"]), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("Number of rehedges")
    plt.ylabel("RMSE of hedge error")
    plt.title("Tradeoff: hedge error vs rehedge count (RMSE)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regime_rehedge_v2_plot_tradeoff_rmse_vs_rehedges.png", dpi=160, bbox_inches="tight")
    plt.close()

    norm_rows = []
    for hedge_type in ["delta", "deltavega"]:
        base = overall.loc[overall["method"] == f"{hedge_type}_every_min"].iloc[0]
        rgm = overall.loc[overall["method"] == f"{hedge_type}_regime"].iloc[0]

        norm_rows.append(
            {
                "label": f"{hedge_type}_every_min",
                "hedge_type": hedge_type,
                "x_rehedge_ratio": 1.0,
                "y_mae_ratio": 1.0,
            }
        )
        norm_rows.append(
            {
                "label": f"{hedge_type}_regime",
                "hedge_type": hedge_type,
                "x_rehedge_ratio": float(rgm["n_rehedges"]) / max(float(base["n_rehedges"]), 1.0),
                "y_mae_ratio": float(rgm["mae"]) / max(float(base["mae"]), 1e-12),
            }
        )

    norm_df = pd.DataFrame(norm_rows)
    norm_df.to_csv(OUT_DIR / "regime_rehedge_v2_normalized_tradeoff.csv", index=False)

    plt.figure(figsize=(7.5, 6.0))
    plt.axvline(1.0, linestyle="--")
    plt.axhline(1.0, linestyle="--")
    plt.scatter(norm_df["x_rehedge_ratio"], norm_df["y_mae_ratio"], s=120)
    for _, row in norm_df.iterrows():
        plt.annotate(row["label"], (row["x_rehedge_ratio"], row["y_mae_ratio"]), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("Rehedge count / every-minute")
    plt.ylabel("MAE / every-minute")
    plt.title("Normalized tradeoff relative to every-minute")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regime_rehedge_v2_plot_normalized_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("\nOverall summary:")
    print(overall.to_string(index=False))
    print("\nReduction vs every-minute:")
    print(reduction.to_string(index=False))

if __name__ == "__main__":
    main()
from __future__ import annotations

import math
import re
from datetime import time
from time import perf_counter
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


# -----------------------------
# Timing helper
# -----------------------------
def log_stage(msg: str, t0: float) -> float:
    now = perf_counter()
    print(f"[TIMER] {msg}: {now - t0:,.2f}s")
    return now


# -----------------------------
# Contract parsing
# -----------------------------
def _safe_to_datetime(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT


def parse_contract_name(name: str) -> Optional[Dict]:
    """Parse e.g. 'SPY US 03/06/26 C640 Equity' -> expiry, cp, K."""
    if not isinstance(name, str):
        return None
    m = re.search(r"(\d{2}/\d{2}/\d{2})\s+([CP])\s*([0-9]+(?:\.[0-9]+)?)", name)
    if not m:
        m = re.search(r"(\d{2}/\d{2}/\d{2})\s+([CP])([0-9]+(?:\.[0-9]+)?)", name)
    if not m:
        return None
    exp = pd.to_datetime(m.group(1), format="%m/%d/%y").date()
    return {
        "underlying": name.split()[0] if len(name.split()) else None,
        "expiry_date": exp,
        "cp": m.group(2),
        "K": float(m.group(3)),
        "raw": name,
    }


# -----------------------------
# Black-Scholes + implied vol
# -----------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(S: float, K: float, r: float, T: float, sigma: float, cp: str) -> float:
    if T <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    if sigma <= 0:
        discK = K * math.exp(-r * T)
        return max(S - discK, 0.0) if cp == "C" else max(discK - S, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if cp == "C":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_vega(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * norm_pdf(d1) * sqrtT


def implied_vol_mid(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    cp: str,
    vol_low: float = 1e-6,
    vol_high: float = 5.0,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> float:
    if not (np.isfinite(price) and np.isfinite(S) and np.isfinite(K) and np.isfinite(T)):
        return float("nan")
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    discK = K * math.exp(-r * T)
    intrinsic = max(S - discK, 0.0) if cp == "C" else max(discK - S, 0.0)
    upper = S if cp == "C" else discK
    if price < intrinsic - 1e-10 or price > upper + 1e-10:
        return float("nan")

    f_low = bs_price(S, K, r, T, vol_low, cp) - price
    f_high = bs_price(S, K, r, T, vol_high, cp) - price

    if f_low == 0.0:
        return vol_low
    if f_high == 0.0:
        return vol_high

    if f_low * f_high > 0:
        vh = vol_high
        for _ in range(8):
            vh *= 1.5
            f2 = bs_price(S, K, r, T, vh, cp) - price
            if f_low * f2 <= 0:
                vol_high = vh
                f_high = f2
                break
        else:
            return float("nan")

    lo, hi = vol_low, vol_high
    flo = f_low
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = bs_price(S, K, r, T, mid, cp) - price
        if abs(fmid) < tol or (hi - lo) < 1e-8:
            return mid
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


# -----------------------------
# Workbook parsing: this file format
# -----------------------------
def parse_sheet_to_quotes(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Each option contract occupies 5 columns:
        Dates | Open | Close | Value | Volume
    Row 0 = contract name
    Row 1 = field labels
    Row 2+ = data
    """
    if df_raw.empty:
        return pd.DataFrame()

    ncols = df_raw.shape[1]
    out = []

    for i in range(0, ncols, 5):
        if i + 4 >= ncols:
            continue

        contract = df_raw.iloc[0, i]
        info = parse_contract_name(contract)
        if info is None:
            continue

        labels = [str(df_raw.iloc[1, j]).strip().lower() for j in range(i, i + 5)]
        expected = ["dates", "open", "close", "value", "volume"]
        if labels != expected:
            continue

        tmp = pd.DataFrame(
            {
                "timestamp": df_raw.iloc[2:, i].apply(_safe_to_datetime).values,
                "open": pd.to_numeric(df_raw.iloc[2:, i + 1], errors="coerce").values,
                "close": pd.to_numeric(df_raw.iloc[2:, i + 2], errors="coerce").values,
                "value": pd.to_numeric(df_raw.iloc[2:, i + 3], errors="coerce").values,
                "volume": pd.to_numeric(df_raw.iloc[2:, i + 4], errors="coerce").values,
            }
        )

        tmp = tmp.dropna(subset=["timestamp", "close"]).copy()
        if tmp.empty:
            continue

        tmp["sheet"] = sheet_name
        tmp["underlying"] = info["underlying"]
        tmp["expiry_date"] = pd.to_datetime(info["expiry_date"])
        tmp["cp"] = info["cp"]
        tmp["K"] = info["K"]

        # No bid/ask in this workbook. Use close as price proxy.
        tmp["last"] = tmp["close"]
        tmp["mid"] = tmp["close"]

        out.append(tmp)

    if not out:
        return pd.DataFrame()

    q = pd.concat(out, ignore_index=True)
    q = q.sort_values(["timestamp", "expiry_date", "cp", "K"]).reset_index(drop=True)
    return q


def parse_workbook_to_quotes(file_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(file_path)
    all_quotes = []

    for sh in tqdm(xls.sheet_names, desc="Reading workbook sheets"):
        df_raw = pd.read_excel(file_path, sheet_name=sh, header=None)
        q = parse_sheet_to_quotes(df_raw, sh)
        if not q.empty:
            all_quotes.append(q)

    if not all_quotes:
        raise ValueError("No option data parsed from workbook.")

    quotes = pd.concat(all_quotes, ignore_index=True)
    quotes["timestamp"] = pd.to_datetime(quotes["timestamp"])
    quotes["expiry_date"] = pd.to_datetime(quotes["expiry_date"]).dt.normalize()
    return quotes


# -----------------------------
# Spot inference from put-call parity
# -----------------------------
def infer_spot_from_parity(
    quotes: pd.DataFrame,
    time_bin: str = "10s",
    min_pair_volume: float = 1.0,
) -> pd.DataFrame:
    """
    Infer S(t) from matched call/put prices with same timestamp-bin, expiry, strike:
        C - P = S - K*exp(-rT), with r=0 => S = C - P + K
    """
    q = quotes.copy()
    q["tbin_spot"] = q["timestamp"].dt.floor(time_bin)

    cols = ["tbin_spot", "expiry_date", "K", "cp", "mid", "volume"]
    qp = q[cols].copy()

    piv_px = (
        qp.pivot_table(
            index=["tbin_spot", "expiry_date", "K"],
            columns="cp",
            values="mid",
            aggfunc="last",
        )
        .reset_index()
    )
    piv_px.columns.name = None

    piv_vol = (
        qp.pivot_table(
            index=["tbin_spot", "expiry_date", "K"],
            columns="cp",
            values="volume",
            aggfunc="sum",
        )
        .reset_index()
    )
    piv_vol.columns.name = None
    piv_vol = piv_vol.rename(columns={"C": "vol_c", "P": "vol_p"})

    pairs = piv_px.merge(piv_vol, on=["tbin_spot", "expiry_date", "K"], how="left")
    if "C" not in pairs.columns:
        pairs["C"] = np.nan
    if "P" not in pairs.columns:
        pairs["P"] = np.nan

    pairs = pairs.dropna(subset=["C", "P"]).copy()
    if pairs.empty:
        raise ValueError("Could not form any call/put parity pairs from workbook.")

    pairs["pair_volume"] = pairs[["vol_c", "vol_p"]].fillna(0).sum(axis=1)
    pairs = pairs[pairs["pair_volume"] >= min_pair_volume].copy()

    pairs["S_parity"] = pairs["C"] - pairs["P"] + pairs["K"]
    pairs = pairs[np.isfinite(pairs["S_parity"]) & (pairs["S_parity"] > 0)].copy()
    if pairs.empty:
        raise ValueError("No valid positive spot estimates from parity.")

    def robust_weighted_spot(g: pd.DataFrame) -> float:
        x = g["S_parity"].values.astype(float)
        w = np.sqrt(np.clip(g["pair_volume"].fillna(1.0).values.astype(float), 1.0, None))
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if np.isfinite(mad) and mad > 0:
            keep = np.abs(x - med) <= 4.0 * 1.4826 * mad
            x = x[keep]
            w = w[keep]
        if len(x) == 0:
            return float("nan")
        return float(np.sum(w * x) / np.sum(w))

    grouped = list(pairs.groupby("tbin_spot"))
    out = []
    for ts, g in tqdm(grouped, total=len(grouped), desc="Inferring spot from parity"):
        out.append({"timestamp": ts, "S": robust_weighted_spot(g)})

    spot = pd.DataFrame(out)
    spot = spot.dropna(subset=["S"]).sort_values("timestamp").reset_index(drop=True)
    return spot


# -----------------------------
# Surface fit helpers
# -----------------------------
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


def weighted_quadratic_fixed_intercept(
    k: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    intercept: float,
) -> Tuple[float, float]:
    k = np.asarray(k, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    m = np.isfinite(k) & np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(intercept)
    k, y, w = k[m], y[m], w[m]
    if len(k) < 5:
        return (float("nan"), float("nan"))

    X = np.vstack([k, k * k]).T
    target = y - intercept
    XtW = X.T * w
    A = XtW @ X
    bvec = XtW @ target

    try:
        beta = np.linalg.solve(A, bvec)
        return float(beta[0]), float(beta[1])
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"))


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

    w = shape_weights_from_df(gg)
    if np.sum(w) <= 1e-12:
        return float(np.nanmedian(gg["iv"].values))

    return float(np.sum(w * gg["iv"].values) / np.sum(w))


def local_curvature_from_quadratic(
    g_shape: pd.DataFrame,
    sigma_atm: float,
) -> float:
    if g_shape.empty or not np.isfinite(sigma_atm):
        return float("nan")

    gg = g_shape.copy()
    gg = gg[np.isfinite(gg["k"]) & np.isfinite(gg["iv"]) & np.isfinite(gg["vega"])].copy()
    if gg.empty:
        return float("nan")

    gg = collapse_to_strikes(gg)
    gg["abs_k"] = np.abs(gg["k"])

    curve_cap = 0.025
    gg = gg[gg["abs_k"] <= curve_cap].copy()
    if gg.empty:
        return float("nan")

    left = gg[gg["k"] < 0].copy()
    right = gg[gg["k"] > 0].copy()

    left_unique = left["K"].nunique()
    right_unique = right["K"].nunique()
    total_unique = gg["K"].nunique()

    if left_unique < 2 or right_unique < 2 or total_unique < 5:
        return float("nan")

    left_span = abs(left["k"].min()) if not left.empty else 0.0
    right_span = right["k"].max() if not right.empty else 0.0
    min_span = 0.0030

    if left_span < min_span or right_span < min_span:
        return float("nan")

    w = shape_weights_from_df(gg)
    if np.sum((w > 0).astype(int)) < 5:
        return float("nan")

    b, c = weighted_quadratic_fixed_intercept(
        gg["k"].values,
        gg["iv"].values,
        w,
        sigma_atm,
    )
    if not np.isfinite(c):
        return float("nan")

    curvature = 2.0 * c
    curvature_cap = 80.0
    if abs(curvature) > curvature_cap:
        return float("nan")

    return float(curvature)


# -----------------------------
# DTE handling: 1DTE only
# -----------------------------
def keep_only_1dte(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["timestamp"]).dt.normalize()
    out["expiry_date"] = pd.to_datetime(out["expiry_date"]).dt.normalize()
    out["days_to_expiry"] = (out["expiry_date"] - out["trade_date"]).dt.days
    out = out[out["days_to_expiry"] == 1].copy()
    out["dte_bucket"] = "1DTE"
    return out


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(
    file_path: str,
    time_bin: str = "1min",
    spot_time_bin: str = "10s",
    asof_tolerance_seconds: int = 90,
    min_price: float = 0.01,
    k_window: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stage_t = perf_counter()
    total_t = perf_counter()

    quotes = parse_workbook_to_quotes(file_path)
    stage_t = log_stage("Workbook parsed", stage_t)

    spot = infer_spot_from_parity(quotes, time_bin=spot_time_bin, min_pair_volume=1.0)
    stage_t = log_stage("Spot inferred", stage_t)

    quotes["expiry_dt"] = quotes["expiry_date"].apply(lambda d: pd.Timestamp.combine(d.date(), time(16, 0)))
    quotes["tau_min"] = (quotes["expiry_dt"] - quotes["timestamp"]).dt.total_seconds() / 60.0
    quotes["tau"] = quotes["tau_min"] / (365.0 * 24.0 * 60.0)

    q = quotes.copy()
    q = q[np.isfinite(q["mid"]) & (q["mid"] >= min_price)].copy()
    q = q[(q["tau_min"] > 0) & (q["tau_min"] < 7 * 24 * 60)].copy()
    q["volume"] = pd.to_numeric(q["volume"], errors="coerce").fillna(0.0)

    q = q.sort_values("timestamp").reset_index(drop=True)
    q = pd.merge_asof(
        q,
        spot.rename(columns={"S": "S_used"}).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=asof_tolerance_seconds),
    )
    q = q.dropna(subset=["S_used"]).copy()
    q = q[q["S_used"] > 0].copy()

    q = keep_only_1dte(q)
    if q.empty:
        raise ValueError("No 1DTE rows found after filtering. Check trade timestamps and expiry dates.")

    stage_t = log_stage("Initial cleaning, spot merge, and 1DTE filter", stage_t)

    lookback_seconds = 120
    max_quote_age_seconds = 90
    spot_tolerance_seconds = 90

    q["tbin"] = q["timestamp"].dt.floor(time_bin)
    t0 = q["tbin"].min()
    t1 = q["tbin"].max()
    minute_grid = pd.date_range(t0, t1, freq=time_bin)

    contracts = q[["expiry_date", "K", "cp"]].drop_duplicates()
    grid = (
        contracts.assign(_k=1)
        .merge(pd.DataFrame({"tbin": minute_grid, "_k": 1}), on="_k", how="inner")
        .drop(columns="_k")
        .sort_values(["tbin", "expiry_date", "K", "cp"])
        .reset_index(drop=True)
    )

    q_sorted = q.sort_values(["timestamp", "expiry_date", "K", "cp"]).reset_index(drop=True)

    q_sync = pd.merge_asof(
        grid,
        q_sorted,
        left_on="tbin",
        right_on="timestamp",
        by=["expiry_date", "K", "cp"],
        direction="backward",
        tolerance=pd.Timedelta(seconds=lookback_seconds),
    )

    if "tbin" not in q_sync.columns:
        if "tbin_x" in q_sync.columns:
            q_sync = q_sync.rename(columns={"tbin_x": "tbin"})
        elif "tbin_y" in q_sync.columns:
            q_sync = q_sync.rename(columns={"tbin_y": "tbin"})
        else:
            raise KeyError(f"Expected tbin (or tbin_x/tbin_y) in columns, got: {list(q_sync.columns)}")

    q_sync = q_sync.dropna(subset=["timestamp"]).copy()
    q_sync = q_sync.rename(columns={"timestamp": "quote_timestamp"})

    q_sync["timestamp"] = q_sync["tbin"]
    q_sync["quote_age_sec"] = (q_sync["timestamp"] - q_sync["quote_timestamp"]).dt.total_seconds()
    q_sync = q_sync[(q_sync["quote_age_sec"] >= 0) & (q_sync["quote_age_sec"] <= max_quote_age_seconds)].copy()

    q_sync = q_sync.drop(columns=["S_used"], errors="ignore")
    q_sync = q_sync.sort_values("timestamp").reset_index(drop=True)

    q_sync = pd.merge_asof(
        q_sync,
        spot.rename(columns={"S": "S_used"}).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=spot_tolerance_seconds),
    )

    q_sync = q_sync.dropna(subset=["S_used"]).copy()
    q_sync = q_sync[q_sync["S_used"] > 0].copy()

    q_sync["expiry_dt"] = q_sync["expiry_date"].apply(lambda d: pd.Timestamp.combine(d.date(), time(16, 0)))
    q_sync["tau_min"] = (q_sync["expiry_dt"] - q_sync["timestamp"]).dt.total_seconds() / 60.0
    q_sync["tau"] = q_sync["tau_min"] / (365.0 * 24.0 * 60.0)

    q_sync = keep_only_1dte(q_sync)
    if q_sync.empty:
        raise ValueError("No 1DTE rows remain after minute synchronization.")

    q = q_sync
    stage_t = log_stage("Minute sync complete", stage_t)

    # IV + vega with progress bar
    r = 0.0
    q["r"] = r
    ivs = []
    vegas = []

    iv_inputs = list(zip(
        q["mid"].values,
        q["S_used"].values,
        q["K"].values,
        q["r"].values,
        q["tau"].values,
        q["cp"].values,
    ))

    for price, S_used, K, r_i, tau_i, cp_i in tqdm(iv_inputs, total=len(iv_inputs), desc="Computing IVs"):
        iv = implied_vol_mid(price, S_used, K, r_i, tau_i, cp_i)
        ivs.append(iv)
        vegas.append(bs_vega(S_used, K, r_i, tau_i, iv) if math.isfinite(iv) else float("nan"))

    q["iv"] = ivs
    q["vega"] = vegas

    q["k"] = np.log(q["K"] / q["S_used"])
    q = q[np.isfinite(q["k"])].copy()
    stage_t = log_stage("IVs and vegas computed", stage_t)

        # -----------------------------
    # Validation block
    # -----------------------------
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)

    # 1) Basic parsed-data checks
    print("\n[1] BASIC DATA CHECKS")
    print("Rows in q:", len(q))
    print("Unique expiries:", sorted(pd.to_datetime(q["expiry_date"]).dt.date.unique()))
    print("Unique option types:", sorted(q["cp"].dropna().unique()))
    print("Strike range:", float(q["K"].min()), "to", float(q["K"].max()))
    print("Price range:", float(q["mid"].min()), "to", float(q["mid"].max()))
    print("Volume range:", float(q["volume"].min()), "to", float(q["volume"].max()))
    print(q[["timestamp", "expiry_date", "cp", "K", "mid", "volume"]].head(10))

    # 2) DTE distribution check
    print("\n[2] DAYS-TO-EXPIRY CHECK")
    dte_check = q.copy()
    dte_check["trade_date"] = pd.to_datetime(dte_check["timestamp"]).dt.normalize()
    dte_check["expiry_date_norm"] = pd.to_datetime(dte_check["expiry_date"]).dt.normalize()
    dte_check["days_to_expiry_check"] = (
        dte_check["expiry_date_norm"] - dte_check["trade_date"]
    ).dt.days
    print(dte_check["days_to_expiry_check"].value_counts(dropna=False).sort_index())

    # 3) Parity spot dispersion check
    print("\n[3] PARITY SPOT DISPERSION CHECK")
    q_val = quotes.copy()
    q_val["tbin_spot"] = q_val["timestamp"].dt.floor(spot_time_bin)

    pairs_val = (
        q_val.pivot_table(
            index=["tbin_spot", "expiry_date", "K"],
            columns="cp",
            values="mid",
            aggfunc="last",
        )
        .reset_index()
    )
    pairs_val.columns.name = None

    if ("C" in pairs_val.columns) and ("P" in pairs_val.columns):
        pairs_val = pairs_val.dropna(subset=["C", "P"]).copy()
        if not pairs_val.empty:
            pairs_val["S_parity"] = pairs_val["C"] - pairs_val["P"] + pairs_val["K"]

            spot_dispersion = (
                pairs_val.groupby("tbin_spot")["S_parity"]
                .agg(["count", "median", "std", "min", "max"])
                .reset_index()
            )

            print("Matched parity pairs:", len(pairs_val))
            print("Spot dispersion summary:")
            print(spot_dispersion[["count", "median", "std", "min", "max"]].describe())

            print("\nSample parity rows:")
            print(pairs_val[["tbin_spot", "expiry_date", "K", "C", "P", "S_parity"]].head(10))
        else:
            print("No matched call-put pairs after dropping NaNs.")
    else:
        print("Could not form both call and put columns for parity validation.")

    # 4) Repricing error check
    print("\n[4] IV REPRICING ERROR CHECK")
    q["model_price"] = [
        bs_price(S, K, r_i, T, iv, cp_i) if np.isfinite(iv) else np.nan
        for S, K, r_i, T, iv, cp_i in zip(
            q["S_used"].values,
            q["K"].values,
            q["r"].values,
            q["tau"].values,
            q["iv"].values,
            q["cp"].values,
        )
    ]
    q["abs_price_err"] = np.abs(q["model_price"] - q["mid"])
    q["rel_price_err"] = q["abs_price_err"] / q["mid"].replace(0, np.nan)

    print(q[["abs_price_err", "rel_price_err"]].describe())

    bad_reprice = q.loc[
        q["abs_price_err"].fillna(0.0) > 1e-4,
        ["timestamp", "expiry_date", "cp", "K", "mid", "model_price", "iv", "abs_price_err"]
    ].head(10)
    if len(bad_reprice) > 0:
        print("\nSample repricing mismatches:")
        print(bad_reprice)
    else:
        print("All repricing errors are near zero.")

    # 5) ATM weighted vs nearest-IV check
    print("\n[5] ATM IV CONSISTENCY CHECK")

    def nearest_atm_iv(g_sub: pd.DataFrame) -> float:
        gg = g_sub.copy()
        gg = gg[np.isfinite(gg["iv"]) & np.isfinite(gg["k"])].copy()
        if gg.empty:
            return float("nan")
        gg["abs_k"] = np.abs(gg["k"])
        gg = gg.sort_values("abs_k")
        if gg.empty:
            return float("nan")
        return float(gg.iloc[0]["iv"])

    atm_check = []
    for (exp_i, tb_i), g_sub in q.groupby(["expiry_date", "tbin"]):
        if g_sub.empty:
            continue
        sigma_weighted = atm_iv_from_nearby(g_sub, k_abs_max=0.015, top_n=6)
        sigma_nearest = nearest_atm_iv(g_sub)
        atm_check.append(
            {
                "timestamp": tb_i,
                "expiry": exp_i,
                "sigma_weighted": sigma_weighted,
                "sigma_nearest": sigma_nearest,
                "diff": sigma_weighted - sigma_nearest,
            }
        )

    atm_check = pd.DataFrame(atm_check)
    if not atm_check.empty:
        print(atm_check[["sigma_weighted", "sigma_nearest", "diff"]].describe())
        print("\nSample ATM comparisons:")
        print(atm_check.head(10))
    else:
        print("No ATM comparison rows available.")

    # 6) IV shape sanity by cross-section sample
    print("\n[6] CROSS-SECTION SHAPE SAMPLE")
    if "tbin" in q.columns and q["tbin"].notna().any():
        tbin_vals = np.sort(q["tbin"].dropna().unique())
        mid_idx = len(tbin_vals) // 2
        t_sample = tbin_vals[mid_idx]
        xsec = q[q["tbin"] == t_sample].copy().sort_values(["expiry_date", "K", "cp"])
        print("Sample tbin:", t_sample)
        print(xsec[["timestamp", "expiry_date", "cp", "K", "S_used", "mid", "iv", "k"]].head(30))
    else:
        print("No tbin values available for cross-section sample.")

    # 7) Time-series roughness check on raw IV points
    print("\n[7] RAW IV ROUGHNESS CHECK")
    rough = (
        q.sort_values(["expiry_date", "K", "cp", "timestamp"])
         .groupby(["expiry_date", "K", "cp"])["iv"]
         .apply(lambda s: s.diff().abs().median())
         .dropna()
    )
    if len(rough) > 0:
        print(rough.describe())
    else:
        print("Not enough repeated time observations for raw-IV roughness check.")

    print("\n" + "=" * 80)
    print("END VALIDATION REPORT")
    print("=" * 80 + "\n")

    feats = []

    drop = {
        "empty_after_iv": 0,
        "too_few_pts_loc": 0,
        "too_few_left_right": 0,
        "total_unique_fail": 0,
        "left_unique_fail": 0,
        "right_unique_fail": 0,
        "left_span_fail": 0,
        "right_span_fail": 0,
        "iv_cap_fail": 0,
        "fit_fail": 0,
        "curvature_support_fail": 0,
        "kept": 0,
    }

    iv_cap = 2.5
    groups = list(q.groupby(["expiry_date", "tbin"]))
    print(f"[INFO] Cross-sections to process: {len(groups):,}")

    for (exp, tb), g_now in tqdm(groups, total=len(groups), desc="Fitting surface features"):
        if g_now.empty:
            continue

        tau_med = float(np.nanmedian(g_now["tau"].values)) if not g_now.empty else float("nan")

        g = g_now[np.isfinite(g_now["iv"]) & np.isfinite(g_now["S_used"]) & (g_now["S_used"] > 0)].copy()
        if g.empty:
            drop["empty_after_iv"] += 1
            continue

        g["k"] = np.log(g["K"] / g["S_used"])
        g["abs_k"] = np.abs(g["k"])
        g = g[np.isfinite(g["k"])].copy()
        g = g[(g["iv"] > 0) & (g["iv"] <= iv_cap)].copy()

        if g.empty:
            drop["iv_cap_fail"] += 1
            continue

        g = collapse_to_strikes(g)

        sigma_atm = atm_iv_from_nearby(
            g,
            k_abs_max=0.015,
            top_n=6,
        )

        if not (np.isfinite(sigma_atm) and 0 < sigma_atm <= iv_cap):
            drop["iv_cap_fail"] += 1
            continue

        skew = float("nan")
        curvature = float("nan")

        shape_cap = 0.035
        atm_band = 0.006

        g_shape = g[g["abs_k"] <= shape_cap].copy()
        if g_shape.empty:
            feats.append(
                {
                    "timestamp": tb,
                    "expiry": exp,
                    "tau": tau_med,
                    "dte_bucket": "1DTE",
                    "sigma_atm": float(sigma_atm),
                    "skew": float("nan"),
                    "curvature": float("nan"),
                    "num_quotes": 0,
                    "shape_ok": False,
                }
            )
            drop["too_few_pts_loc"] += 1
            drop["kept"] += 1
            continue

        g_left = g_shape[g_shape["k"] < 0].sort_values("abs_k").copy()
        g_right = g_shape[g_shape["k"] > 0].sort_values("abs_k").copy()
        g_atm = g_shape[g_shape["abs_k"] <= atm_band].sort_values("abs_k").head(2).copy()

        n_lin_each_side = 10

        g_lin = (
            pd.concat(
                [
                    g_left.head(n_lin_each_side),
                    g_right.head(n_lin_each_side),
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

        skew_min_total = 6
        skew_min_left = 2
        skew_min_right = 2
        skew_min_span = 0.0025

        skew_ok = (
            total_unique >= skew_min_total
            and left_unique >= skew_min_left
            and right_unique >= skew_min_right
            and left_span >= skew_min_span
            and right_span >= skew_min_span
        )

        if skew_ok:
            w_lin = shape_weights_from_df(g_lin)
            b_lin = weighted_slope_fixed_intercept(
                g_lin["k"].values,
                g_lin["iv"].values,
                w_lin,
                sigma_atm,
            )
            if np.isfinite(b_lin):
                skew = float(b_lin)
            else:
                drop["fit_fail"] += 1
        else:
            drop["too_few_left_right"] += 1
            if total_unique < skew_min_total:
                drop["total_unique_fail"] += 1
            if left_unique < skew_min_left:
                drop["left_unique_fail"] += 1
            if right_unique < skew_min_right:
                drop["right_unique_fail"] += 1
            if left_span < skew_min_span:
                drop["left_span_fail"] += 1
            if right_span < skew_min_span:
                drop["right_span_fail"] += 1

        curve_cap = 0.025
        g_curve = g_shape[g_shape["abs_k"] <= curve_cap].copy()

        curv = local_curvature_from_quadratic(
            g_curve,
            sigma_atm=sigma_atm,
        )

        if np.isfinite(curv):
            curvature = float(curv)
        else:
            drop["curvature_support_fail"] += 1

        feats.append(
            {
                "timestamp": tb,
                "expiry": exp,
                "tau": tau_med,
                "dte_bucket": "1DTE",
                "sigma_atm": float(sigma_atm),
                "skew": float(skew),
                "curvature": float(curvature),
                "num_quotes": int(len(g_shape)),
                "shape_ok": bool(np.isfinite(skew)),
            }
        )
        drop["kept"] += 1

    print("DROP REASONS:", drop)
    stage_t = log_stage("Feature fitting complete", stage_t)

    feat = pd.DataFrame(feats).sort_values(["expiry", "timestamp"]).reset_index(drop=True)

        # -----------------------------
    # Feature validation block
    # -----------------------------
    print("\n" + "=" * 80)
    print("FEATURE VALIDATION REPORT")
    print("=" * 80)

    if not feat.empty:
        print("\n[1] FEATURE COVERAGE")
        print(
            feat.assign(
                has_atm=feat["sigma_atm"].notna(),
                has_skew=feat["skew"].notna(),
                has_curvature=feat["curvature"].notna(),
            )
            .groupby("dte_bucket")[["has_atm", "has_skew", "has_curvature"]]
            .mean()
        )

        print("\n[2] FEATURE SUMMARY")
        cols_show = [c for c in ["sigma_atm", "skew", "curvature", "num_quotes"] if c in feat.columns]
        print(feat[cols_show].describe())

        print("\n[3] FEATURE ROUGHNESS")
        for col in ["sigma_atm", "skew", "curvature"]:
            if col in feat.columns and feat[col].notna().sum() > 1:
                series = feat.sort_values("timestamp")[col]
                print(
                    col,
                    {
                        "std": float(series.std()),
                        "median_abs_diff": float(series.diff().abs().median()),
                        "p95_abs_diff": float(series.diff().abs().quantile(0.95)),
                    },
                )

        print("\n[4] EXTREME FEATURE ROWS")
        for col in ["sigma_atm", "skew", "curvature"]:
            if col in feat.columns and feat[col].notna().any():
                print(f"\nTop 5 abs({col}) rows:")
                tmp = feat.loc[feat[col].notna(), ["timestamp", "expiry", "dte_bucket", col, "num_quotes"]].copy()
                tmp["abs_val"] = tmp[col].abs()
                print(tmp.sort_values("abs_val", ascending=False).head(5).drop(columns="abs_val"))
    else:
        print("Feature table is empty.")

    print("\n" + "=" * 80)
    print("END FEATURE VALIDATION REPORT")
    print("=" * 80 + "\n")

    feat["date"] = pd.to_datetime(feat["timestamp"]).dt.date
    support_summary = (
        feat.assign(
            has_skew=feat["skew"].notna(),
            has_curvature=feat["curvature"].notna(),
            has_atm=feat["sigma_atm"].notna(),
        )
        .groupby(["date", "dte_bucket"])
        .agg(
            rows=("timestamp", "size"),
            atm_rows=("has_atm", "sum"),
            skew_rows=("has_skew", "sum"),
            curvature_rows=("has_curvature", "sum"),
            median_num_quotes=("num_quotes", "median"),
        )
        .reset_index()
    )
    print(support_summary)

    log_stage("Total pipeline runtime", total_t)
    return q, feat


def plot_timeseries(
    feat: pd.DataFrame,
    bucket: str,
    out_prefix: str,
    feature_cols: tuple[str, ...] = ("sigma_atm", "skew", "curvature"),
    n_days: int = 2,
    random_state: int = 42,
    gap_minutes: float = 30.0,
):
    d = feat[feat["dte_bucket"] == bucket].copy()
    if d.empty:
        return []

    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["date"] = d["timestamp"].dt.date

    unique_days = sorted(d["date"].dropna().unique())
    if len(unique_days) == 0:
        return []

    rng = np.random.default_rng(random_state)
    if len(unique_days) <= n_days:
        selected_days = list(unique_days)
    else:
        selected_days = sorted(rng.choice(unique_days, size=n_days, replace=False))

    label_map = {
        "sigma_atm": "ATM IV",
        "skew": "Skew dσ/dk|0",
        "curvature": "Curvature d²σ/dk²|0",
    }

    def _break_gaps(ts: pd.Series, y: pd.Series, local_gap_minutes: float):
        ts = pd.to_datetime(ts).values.astype("datetime64[ns]")
        y = np.asarray(y, dtype=float)
        if len(ts) == 0:
            return ts, y

        out_t = [ts[0]]
        out_y = [y[0]]

        for i in range(1, len(ts)):
            dt_min = (ts[i] - ts[i - 1]) / np.timedelta64(1, "m")
            if dt_min > local_gap_minutes:
                out_t.append(np.datetime64("NaT"))
                out_y.append(np.nan)
            out_t.append(ts[i])
            out_y.append(y[i])

        return np.array(out_t), np.array(out_y)

    written_files = []

    for day in selected_days:
        dd = d[d["date"] == day].copy()
        if dd.empty:
            continue

        for col in feature_cols:
            if col not in dd.columns:
                continue

            plt.figure(figsize=(11, 4))
            plotted_any = False

            group_col = "expiry" if "expiry" in dd.columns else ("expiry_date" if "expiry_date" in dd.columns else None)

            if group_col is None:
                g = dd.sort_values("timestamp")
                g_valid = g[g[col].notna()].copy()
                if not g_valid.empty:
                    local_gap = 5.0 if col == "curvature" else gap_minutes
                    x_plot, y_plot = _break_gaps(g_valid["timestamp"], g_valid[col], local_gap)
                    plt.plot(x_plot, y_plot, linewidth=1.5)
                    plotted_any = True
            else:
                for exp, g in dd.groupby(group_col):
                    g = g.sort_values("timestamp")
                    g_valid = g[g[col].notna()].copy()
                    if g_valid.empty:
                        continue

                    local_gap = 5.0 if col == "curvature" else gap_minutes
                    x_plot, y_plot = _break_gaps(g_valid["timestamp"], g_valid[col], local_gap)

                    plt.plot(
                        x_plot,
                        y_plot,
                        linewidth=1.5,
                        label=str(pd.to_datetime(exp).date()) if pd.notna(exp) else "expiry",
                    )
                    plotted_any = True

            if not plotted_any:
                plt.close()
                continue

            plt.xlabel("Time")
            plt.ylabel(label_map.get(col, col))
            plt.title(f"{bucket} {label_map.get(col, col)} vs time — {day}")

            if group_col is not None:
                plt.legend(loc="best", fontsize=8)

            plt.tight_layout()

            out_file = f"{out_prefix}_{bucket}_{col}_{day}.png"
            plt.savefig(out_file, dpi=140)
            plt.close()

            written_files.append(out_file)

    return selected_days


def plot_snapshot(
    quotes_iv: pd.DataFrame,
    selected_days,
    k_window: float,
    out_prefix: str,
):
    q = quotes_iv.copy()
    if q.empty:
        return

    q["timestamp"] = pd.to_datetime(q["timestamp"])
    q["trade_date"] = pd.to_datetime(q["trade_date"]).dt.date if "trade_date" in q.columns else q["timestamp"].dt.date

    q = q[np.isfinite(q["iv"])]
    q = q[q["dte_bucket"] == "1DTE"].copy()
    if q.empty:
        return

    written_files = []

    for i, day in enumerate(selected_days, start=1):
        day_q = q[q["trade_date"] == day].copy()
        if day_q.empty:
            continue

        counts = (
            day_q.groupby(["tbin", "expiry_date"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        if counts.empty:
            continue

        ts = counts.iloc[0]["tbin"]
        exp = counts.iloc[0]["expiry_date"]

        snap = day_q[(day_q["tbin"] == ts) & (day_q["expiry_date"] == exp)].copy()
        if snap.empty:
            continue

        snap["k"] = np.log(snap["K"] / snap["S_used"])
        snap["abs_k"] = np.abs(snap["k"])
        snap = snap[np.isfinite(snap["k"])].copy()
        snap = snap[(snap["iv"] > 0) & (snap["iv"] <= 2.5)].copy()

        snap = collapse_to_strikes(snap)

        local_k = min(k_window, 0.035)
        snap_loc = snap[np.abs(snap["k"]) <= local_k].copy()
        if len(snap_loc) < 5:
            continue

        sigma_atm = atm_iv_from_nearby(snap, k_abs_max=0.015, top_n=6)
        if not np.isfinite(sigma_atm):
            continue

        w = shape_weights_from_df(snap_loc)
        b, c = weighted_quadratic_fixed_intercept(
            snap_loc["k"].values,
            snap_loc["iv"].values,
            w,
            sigma_atm,
        )
        if not (np.isfinite(b) and np.isfinite(c)):
            continue

        curvature = 2.0 * c

        k_grid = np.linspace(snap_loc["k"].min(), snap_loc["k"].max(), 200)
        iv_fit = sigma_atm + b * k_grid + c * (k_grid ** 2)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(snap["k"], snap["iv"], s=20, label="Observed IV")
        ax.plot(
            k_grid,
            iv_fit,
            linewidth=2.0,
            label=f"Fit: σ={sigma_atm:.4f}, skew={b:.4f}, curv={curvature:.4f}",
        )
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("k = ln(K/S)")
        ax.set_ylabel("Implied vol")
        ax.set_title(f"1DTE IV snapshot — {day} — {pd.to_datetime(ts)}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        out_file = f"{out_prefix}_{i}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

        written_files.append(out_file)

    return written_files


def main(options_data_path : str = "MATH_86_Kason_2.xlsx", out_put_path : str = '/'):
    options_xlsx_path = options_data_path
    output_dir = out_put_path
    os.makedirs(output_dir, exist_ok=True)

    quotes_iv, feat = run_pipeline(
        file_path=options_xlsx_path,
        time_bin="1min",
        spot_time_bin="10s",
        asof_tolerance_seconds=90,
        k_window=0.15,
    )

    cleaned_cols = [
        "timestamp", "expiry_date", "cp", "K",
        "open", "close", "value", "volume",
        "last", "mid", "tau_min", "tau",
        "S_used", "r", "iv", "vega", "k",
        "trade_date", "days_to_expiry", "dte_bucket",
        "tbin", "quote_timestamp", "quote_age_sec",
        "model_price", "abs_price_err", "rel_price_err",
    ]
    cleaned_cols = [c for c in cleaned_cols if c in quotes_iv.columns]

    cleaned_quotes_file_csv = os.path.join(output_dir, "cleaned_quotes_with_iv_1dte.csv")
    features_file_csv = os.path.join(output_dir, "iv_surface_features_1dte.csv")

    cleaned_quotes_out = quotes_iv[cleaned_cols].copy()
    cleaned_quotes_out.to_csv(cleaned_quotes_file_csv, index=False)
    feat.to_csv(features_file_csv, index=False)

    print(f"[FILE WRITTEN] {cleaned_quotes_file_csv}")
    print(f"[FILE WRITTEN] {features_file_csv}")

    selected_days = plot_timeseries(
        feat,
        "1DTE",
        os.path.join(output_dir, "timeseries"),
        feature_cols=("sigma_atm", "skew", "curvature"),
        n_days=2,
        random_state=42,
    )

    plot_snapshot(
        quotes_iv,
        selected_days,
        0.15,
        os.path.join(output_dir, "iv_surface_snapshot"),
    )

if __name__ == "__main__":


    output_dir = "/Users/brendonbazzani/VS Code Projects-python"

    files_to_delete = [
        "cleaned_quotes_with_iv_1dte.csv",
        "iv_surface_features_1dte.csv",
        "iv_surface_snapshot_1.png",
        "iv_surface_snapshot_2.png",
    ]

    for fn in files_to_delete:
        path = os.path.join(output_dir, fn)
        if os.path.exists(path):
            os.remove(path)

    for pattern in [
        "timeseries_1DTE_sigma_atm_*.png",
        "timeseries_1DTE_skew_*.png",
        "timeseries_1DTE_curvature_*.png",
    ]:
        for file in glob.glob(os.path.join(output_dir, pattern)):
            os.remove(file)

    main()
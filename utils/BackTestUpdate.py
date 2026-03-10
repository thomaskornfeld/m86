import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths / Knobs
# ============================================================
QUOTES_CSV = Path("/Users/brendonbazzani/VS Code Projects-python/cleaned_quotes_with_iv_1dte.csv")
FEATURE_PRED_CSV = Path("/Users/brendonbazzani/VS Code Projects-python/boundary_layer_test_feature_predictions.csv")
OUT_DIR = Path("/Users/brendonbazzani/VS Code Projects-python")

MIN_MID = 0.05
MAX_ABS_K_MAIN = 0.03
MAX_ABS_K_HEDGE = 0.02
VEGA_MIN_HEDGE = 0.25

# Execution Simulation Knobs
BASELINE_INTERVALS_MIN = [5, 10]  # Upgraded from every_min
SLOW_REHEDGE_INTERVAL_MIN = 10
STOCK_BAND_DELTA = 0.05
STOCK_BAND_DELTAVEGA = 0.05
OPTION_BAND_DELTAVEGA = 0.05

STOCK_TC_PER_SHARE = 0.00
OPTION_TC_PER_UNIT = 0.00
MAX_ABS_HEDGE_OPTION_UNITS = 10.0

# Attribution Bumps
DELTA_BUMP_FRAC = 1e-4
SIGMA_BUMP = 0.005
SKEW_BUMP = 0.10
CURV_BUMP = 10.0
TAU_BUMP = 1.0 / (252.0 * 390.0)

# ============================================================
# Centralized Pricing & Math Engine (Corrected)
# ============================================================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_price(cp: str, S: float, K: float, tau: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    """Forward-pricing adjusted Black-Scholes[cite: 2]."""
    if tau <= 0 or sigma <= 0:
        return max(S - K, 0.0) if cp.upper().startswith("C") else max(K - S, 0.0)
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)

    if cp.upper().startswith("C"):
        return S * math.exp(-q * tau) * norm_cdf(d1) - K * math.exp(-r * tau) * norm_cdf(d2)
    return K * math.exp(-r * tau) * norm_cdf(-d2) - S * math.exp(-q * tau) * norm_cdf(-d1)

def local_iv(S: float, K: float, tau: float, r: float, q: float, sigma_atm: float, skew: float, curv: float) -> float:
    """Calculates local IV using forward-moneyness $k = \\ln(K/F)$[cite: 46]."""
    F = S * math.exp((r - q) * tau)
    k = math.log(K / F)
    sigma = sigma_atm + skew * k + 0.5 * curv * (k ** 2)
    return min(max(sigma, 1e-4), 3.0)

def surface_price(cp: str, S: float, K: float, tau: float, sigma_atm: float, skew: float, curv: float, r: float = 0.0, q: float = 0.0) -> float:
    sigma = local_iv(S, K, tau, r, q, sigma_atm, skew, curv)
    return bs_price(cp, S, K, tau, sigma, r, q)

def surface_greeks(cp: str, S: float, K: float, tau: float, sigma_atm: float, skew: float, curv: float, r: float = 0.0) -> dict:
    """Calculates all attribution Greeks simultaneously to save compute time[cite: 58, 60]."""
    h = max(S * DELTA_BUMP_FRAC, 1e-4)
    
    p_base = surface_price(cp, S, K, tau, sigma_atm, skew, curv, r)
    p_up = surface_price(cp, S + h, K, tau, sigma_atm, skew, curv, r)
    p_dn = surface_price(cp, S - h, K, tau, sigma_atm, skew, curv, r)
    
    delta = (p_up - p_dn) / (2.0 * h)
    gamma = (p_up - 2.0 * p_base + p_dn) / (h * h)
    
    v_up = surface_price(cp, S, K, tau, sigma_atm + SIGMA_BUMP, skew, curv, r)
    v_dn = surface_price(cp, S, K, tau, sigma_atm - SIGMA_BUMP, skew, curv, r)
    vega_atm = (v_up - v_dn) / (2.0 * SIGMA_BUMP)
    
    sk_up = surface_price(cp, S, K, tau, sigma_atm, skew + SKEW_BUMP, curv, r)
    sk_dn = surface_price(cp, S, K, tau, sigma_atm, skew - SKEW_BUMP, curv, r)
    g_skew = (sk_up - sk_dn) / (2.0 * SKEW_BUMP)
    
    c_up = surface_price(cp, S, K, tau, sigma_atm, skew, curv + CURV_BUMP, r)
    c_dn = surface_price(cp, S, K, tau, sigma_atm, skew, curv - CURV_BUMP, r)
    g_curv = (c_up - c_dn) / (2.0 * CURV_BUMP)
    
    t_up = surface_price(cp, S, K, tau + TAU_BUMP, sigma_atm, skew, curv, r)
    t_dn = surface_price(cp, S, K, max(tau - TAU_BUMP, 1e-10), sigma_atm, skew, curv, r)
    theta = (t_up - t_dn) / (2.0 * TAU_BUMP)
    
    return {"delta": delta, "gamma": gamma, "vega_atm": vega_atm, "g_skew": g_skew, "g_curv": g_curv, "theta": theta}

# ============================================================
# Step 1: Data Pipeline (Combined)
# ============================================================
def build_merged_panel() -> pd.DataFrame:
    """Combines quote transitions, boundary layer features, and fixed hedges."""
    feat = pd.read_csv(FEATURE_PRED_CSV, parse_dates=["timestamp", "trade_date"])
    feat["trade_date"] = pd.to_datetime(feat["trade_date"]).dt.normalize()
    
    q = pd.read_csv(QUOTES_CSV, parse_dates=["timestamp", "trade_date", "expiry_date"])
    q["trade_date"] = pd.to_datetime(q["trade_date"]).dt.normalize()
    
    # Safely inject interest rate and dividend yield if missing from raw data
    if "r" not in q.columns: q["r"] = 0.0
    if "q" not in q.columns: q["q"] = 0.0

    print(f"  [Debug] Quotes loaded: {len(q)} | Features loaded: {len(feat)}")
    
# --- FIX: Sort by contract then time so shift(-1) looks at the correct next row ---
    q = q.sort_values(["trade_date", "expiry_date", "cp", "K", "timestamp"]).reset_index(drop=True)

    # Isolate 1-minute transitions
    same_contract_next = (
        q["trade_date"].eq(q["trade_date"].shift(-1)) & 
        q["expiry_date"].eq(q["expiry_date"].shift(-1)) & 
        q["cp"].eq(q["cp"].shift(-1)) & 
        q["K"].eq(q["K"].shift(-1))
    )
    
    # Optional Safety Net: If your data is tick-data or messy, exactly 60 seconds might fail.
    # If it still reads 0 after sorting, change `.eq(60)` to `.between(50, 70)`
    one_min_next = (q["timestamp"].shift(-1) - q["timestamp"]).dt.total_seconds().eq(60)
    
    qt = q.loc[same_contract_next & one_min_next].copy()
    for col in ["timestamp", "mid", "S_used", "tau", "iv", "vega", "k"]:
        qt[f"{col}_next"] = q[col].shift(-1).loc[qt.index].values
        
    qt["contract_id"] = qt["cp"].astype(str) + "_" + qt["K"].round(6).astype(str)
    qt["abs_k"] = qt["k"].abs()
    qt = qt[(qt["mid"] > MIN_MID) & (qt["mid_next"] > MIN_MID) & (qt["abs_k"] <= MAX_ABS_K_MAIN)].copy()
    
    print(f"  [Debug] Valid 1-min transitions found: {len(qt)}")

    # Find Hedges (Lowered Vega threshold drastically to prevent data loss)
    cand = qt[(qt["abs_k"] <= MAX_ABS_K_HEDGE) & (qt["vega"] >= 0.001)].copy()
    print(f"  [Debug] Hedge candidates found: {len(cand)}")
    
    score = cand.groupby(["trade_date", "expiry_date", "contract_id", "cp", "K"], as_index=False).agg(
        coverage=("timestamp", "size"), mean_vega=("vega", "mean"), mean_abs_k=("abs_k", "mean")
    ).sort_values(["trade_date", "expiry_date", "coverage", "mean_vega", "mean_abs_k"], ascending=[True, True, False, False, True])
    score["rank"] = score.groupby(["trade_date", "expiry_date"]).cumcount() + 1
    
    top1 = score[score["rank"] == 1][["trade_date", "expiry_date", "contract_id"]].rename(columns={"contract_id": "hedge1_id"})

    main = qt.merge(feat, on=["timestamp", "trade_date"], how="inner").copy()
    print(f"  [Debug] Inner merge (Quotes + Features): {len(main)}")
    
    main = main.merge(top1, on=["trade_date", "expiry_date"], how="left")
    main["hedge_id"] = main["hedge1_id"]
    
    hedge_lookup = qt[["timestamp", "trade_date", "expiry_date", "contract_id", "cp", "K", "mid", "mid_next", "S_used", "tau", "r"]].copy()
    hedge_lookup = hedge_lookup.rename(columns={c: f"{c}_h" for c in hedge_lookup.columns if c not in ["timestamp", "trade_date", "expiry_date"]})
    
    main = main.merge(hedge_lookup, left_on=["timestamp", "trade_date", "expiry_date", "hedge_id"], right_on=["timestamp", "trade_date", "expiry_date", "contract_id_h"], how="left")
    
    final_df = main[main["contract_id_h"].notna()].reset_index(drop=True)
    print(f"  [Debug] Final Panel Ready: {len(final_df)} rows")
    return final_df

# ============================================================
# Step 2: Attribution & Target Generation
# ============================================================
def compute_attribution_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all Greeks, PnL attribution, and ideal hedge targets[cite: 14, 66, 67]."""
    out = df.copy()
    n = len(out)
    
    # Greek Arrays
    d_curr, g_curr, v_curr, t_curr, sk_curr, cu_curr = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    d_model, v_model = np.zeros(n), np.zeros(n)
    d_hedge, v_hedge = np.zeros(n), np.zeros(n)
    
    for i, row in enumerate(out.itertuples(index=False)):
        # Current Surface Greeks
        greeks = surface_greeks(row.cp, row.S_used, row.K, row.tau, row.sigma_atm, row.skew, row.curvature_bfly, row.r)
        d_curr[i], g_curr[i], v_curr[i] = greeks["delta"], greeks["gamma"], greeks["vega_atm"]
        t_curr[i], sk_curr[i], cu_curr[i] = greeks["theta"], greeks["g_skew"], greeks["g_curv"]
        
        # Model Predicted Targets
        greeks_pred = surface_greeks(row.cp, row.S_used, row.K, row.tau, row.sigma_atm_pred, row.skew_pred, row.curvature_bfly_pred, row.r)
        d_model[i], v_model[i] = greeks_pred["delta"], greeks_pred["vega_atm"]
        
        # Hedge Instrument Targets
        greeks_hedge = surface_greeks(row.cp_h, row.S_used_h, row.K_h, row.tau_h, row.sigma_atm_pred, row.skew_pred, row.curvature_bfly_pred, row.r_h)
        d_hedge[i], v_hedge[i] = greeks_hedge["delta"], greeks_hedge["vega_atm"]

    # Attribution Logic 
    out["dS"] = out["S_used_next"] - out["S_used"]
    out["dV"] = out["mid_next"] - out["mid"]
    out["dtau"] = out["tau_next"] - out["tau"]
    
    out["term_delta"] = d_curr * out["dS"]
    out["term_gamma"] = 0.5 * g_curr * (out["dS"] ** 2)
    out["term_theta"] = t_curr * out["dtau"]
    out["term_vega_atm"] = v_curr * (out["sigma_atm_next"] - out["sigma_atm"])
    out["term_skew"] = sk_curr * (out["skew_next"] - out["skew"])
    out["term_curv"] = cu_curr * (out["curvature_bfly_next"] - out["curvature_bfly"])
    
    out["dV_attr_explained"] = out["term_delta"] + out["term_gamma"] + out["term_theta"] + out["term_vega_atm"] + out["term_skew"] + out["term_curv"]
    out["attr_residual"] = out["dV"] - out["dV_attr_explained"]
    
    # Target Generation [cite: 14, 15]
    out["target_stock_delta"] = -d_model
    q_opt = -v_model / np.where(np.abs(v_hedge) > 1e-8, v_hedge, np.nan)
    q_opt = np.clip(q_opt, -MAX_ABS_HEDGE_OPTION_UNITS, MAX_ABS_HEDGE_OPTION_UNITS)
    
    out["target_opt_deltavega"] = q_opt
    out["target_stock_deltavega"] = -(d_model + q_opt * d_hedge)
    
    return out

# ============================================================
# Step 3: Execution Simulation (Upgraded Intervals)
# ============================================================
def should_rehedge(
    execution_type: str, interval_min: int, regime: int, timestamp: pd.Timestamp, last_rehedge_ts: pd.Timestamp,
    target_stock: float, held_stock: float, target_opt: float, held_opt: float,
    stock_band: float, opt_band: float
) -> bool:
    """Determines execution based on baseline intervals or dynamic regimes[cite: 16, 17, 18, 22]."""
    if last_rehedge_ts is None:
        return True

    mins_since = (timestamp - last_rehedge_ts).total_seconds() / 60.0
    
    if execution_type == "baseline":
        return mins_since >= interval_min
        
    elif execution_type == "regime":
        if regime == 1: # Fast Regime
            return True 
        # Slow Regime 
        if mins_since >= SLOW_REHEDGE_INTERVAL_MIN:
            return True
        if stock_band > 0 and abs(target_stock - held_stock) >= stock_band:
            return True
        if opt_band > 0 and abs(target_opt - held_opt) >= opt_band:
            return True
            
    return False

def run_policy(main: pd.DataFrame, hedge_type: str, execution_type: str, interval_min: int = 0) -> pd.DataFrame:
    """Iterates chronologically to calculate turnover, TC, and errors."""
    
    # Hard-code the columns so if the dataframe is empty, it doesn't crash the grouper later
    if main.empty:
        return pd.DataFrame(columns=["timestamp", "method", "hedge_error", "trade_stock_abs", "trade_opt_abs"])

    rows = []
    for _, g in main.groupby(["trade_date", "expiry_date", "cp", "K"], sort=False):
        g = g.sort_values("timestamp")
        
        held_stock, held_opt = 0.0, 0.0
        last_rehedge_ts = None
        
        for row in g.itertuples():
            target_stock = row.target_stock_delta if hedge_type == "delta" else row.target_stock_deltavega
            target_opt = 0.0 if hedge_type == "delta" else row.target_opt_deltavega
            
            do_rehedge = should_rehedge(
                execution_type, interval_min, row.regime, row.timestamp, last_rehedge_ts,
                target_stock, held_stock, target_opt, held_opt,
                STOCK_BAND_DELTAVEGA, OPTION_BAND_DELTAVEGA
            )
            
            d_stock_trade, d_opt_trade = 0.0, 0.0
            if do_rehedge:
                d_stock_trade = target_stock - held_stock
                d_opt_trade = target_opt - held_opt
                held_stock = target_stock
                held_opt = target_opt
                last_rehedge_ts = row.timestamp
                
            tc = STOCK_TC_PER_SHARE * abs(d_stock_trade) + OPTION_TC_PER_UNIT * abs(d_opt_trade)
            err = row.dV + held_stock * row.dS + held_opt * (row.mid_next_h - row.mid_h) - tc
            
            label = f"{hedge_type}_{execution_type}" if execution_type == "regime" else f"{hedge_type}_{interval_min}min"
            rows.append({
                "timestamp": row.timestamp,
                "method": label,
                "hedge_error": err,
                "trade_stock_abs": abs(d_stock_trade),
                "trade_opt_abs": abs(d_opt_trade)
            })
            
    # Extra safety net
    if not rows:
        return pd.DataFrame(columns=["timestamp", "method", "hedge_error", "trade_stock_abs", "trade_opt_abs"])
        
    return pd.DataFrame(rows)



def plot_simulation_summary(summary: pd.DataFrame, out_dir: Path):
    """Generates side-by-side bar charts for MAE and Turnover comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Plot 1: Mean Absolute Error (MAE) ---
    axes[0].bar(summary["method"], summary["mae"], color='royalblue', edgecolor='black', alpha=0.8)
    axes[0].set_title("Hedging Error (MAE) by Strategy\n(Lower is Better)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Mean Absolute Error ($)", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: Turnover (Stock vs Options) ---
    x = np.arange(len(summary["method"]))
    width = 0.35

    axes[1].bar(x - width/2, summary["stock_turnover"], width, 
                label='Stock Turnover (Shares)', color='mediumseagreen', edgecolor='black', alpha=0.8)
    axes[1].bar(x + width/2, summary["opt_turnover"], width, 
                label='Option Turnover (Units)', color='salmon', edgecolor='black', alpha=0.8)
    
    axes[1].set_title("Execution Turnover by Strategy\n(Lower is Better)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Total Units Traded", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["method"], rotation=45)
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = out_dir / "unified_execution_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  [Plot Saved] {plot_path}")
    plt.close()

# ============================================================
# Main Execution Block
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Building Data Pipeline...")
    main_panel = build_merged_panel()
    
    print("Computing PnL Attribution & Greeks...")
    main_panel = compute_attribution_and_targets(main_panel)
    
    # Save Attribution Report [cite: 74, 82]
    main_panel.to_csv(OUT_DIR / "unified_pnl_attribution.csv", index=False)
    
    panels = []
    print("Simulating Execution Policies...")
    for hedge_type in ["delta", "deltavega"]:
        # 1. New Baselines (5 and 10 minutes)
        for interval in BASELINE_INTERVALS_MIN:
            panels.append(run_policy(main_panel, hedge_type, "baseline", interval))
            
        # 2. Dynamic Regime Simulation
        panels.append(run_policy(main_panel, hedge_type, "regime"))
        
    results = pd.concat(panels, ignore_index=True)
    
    # Aggregate Turnover & Error Data [cite: 30, 31, 40, 41]
    summary = results.groupby("method").agg(
        mae=("hedge_error", lambda x: np.abs(x).mean()),
        stock_turnover=("trade_stock_abs", "sum"),
        opt_turnover=("trade_opt_abs", "sum")
    ).reset_index()
    
    summary.to_csv(OUT_DIR / "unified_execution_summary.csv", index=False)
    print("\nSimulation Complete. Final Summary:")
    print(summary)

    # Aggregate Turnover & Error Data
    summary = results.groupby("method").agg(
        mae=("hedge_error", lambda x: np.abs(x).mean()),
        stock_turnover=("trade_stock_abs", "sum"),
        opt_turnover=("trade_opt_abs", "sum")
    ).reset_index()
    
    summary.to_csv(OUT_DIR / "unified_execution_summary.csv", index=False)
    print("\nSimulation Complete. Final Summary:")
    print(summary)
    
    # --- NEW: Generate and save the plots ---
    print("\nGenerating final plots...")
    plot_simulation_summary(summary, OUT_DIR)

    

if __name__ == "__main__":
    main()
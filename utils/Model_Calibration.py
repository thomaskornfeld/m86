import math
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# User paths / knobs
# ============================================================

FEATURES_FILE = "regime_labeled_intraday_features_1dte.csv"
DEFAULT_INPUT_DIR = Path("Regime_Out")
DEFAULT_OUTPUT_DIR = Path("Calibration_Out")

TRAIN_DAYS = 16
VAL_DAYS = 4

# candidate shock half-lives for Y_t = phi Y_{t-1} + J_t
SHOCK_HALFLIFE_GRID_MIN = [1, 2, 3, 5, 8, 10, 15, 20]

# smile curvature is noisier than level/skew, so forecast it with heavier clipping
CURV_CLIP = 800.0


# ============================================================
# Feature panel / model fit
# ============================================================
def load_feature_panel(path: Path) -> pd.DataFrame:
    feat = pd.read_csv(path, parse_dates=["timestamp", "trade_date"])
    feat["trade_date"] = pd.to_datetime(feat["trade_date"]).dt.normalize()
    feat = feat.sort_values(["trade_date", "timestamp"]).reset_index(drop=True)

    feat["minute_of_day"] = feat["timestamp"].dt.hour * 60 + feat["timestamp"].dt.minute
    feat["v_atm"] = feat["sigma_atm"] ** 2

    if "fast_trigger" in feat.columns:
        feat["event_impulse"] = feat["fast_trigger"].astype(float)
    else:
        feat["event_impulse"] = feat["event_any"].astype(float)

    return feat


def make_feature_transitions(feat: pd.DataFrame) -> pd.DataFrame:
    same_day_next = feat["trade_date"].eq(feat["trade_date"].shift(-1))
    one_min_next = (feat["timestamp"].shift(-1) - feat["timestamp"]).dt.total_seconds().eq(60)

    trans = feat.loc[same_day_next & one_min_next].copy()
    keep = [
        "timestamp",
        "trade_date",
        "minute_of_day",
        "regime",
        "regime_name",
        "event_impulse",
        "event_score",
        "sigma_atm",
        "skew",
        "curvature_bfly",
        "v_atm",
    ]
    trans = trans[keep].copy()

    for col in ["timestamp", "minute_of_day", "sigma_atm", "skew", "curvature_bfly", "v_atm"]:
        trans[f"{col}_next"] = feat[col].shift(-1).loc[trans.index].values

    trans = trans.reset_index(drop=True)
    return trans


def split_dates(dates: list[pd.Timestamp], train_days: int = TRAIN_DAYS, val_days: int = VAL_DAYS):
    train_dates = set(dates[:train_days])
    val_dates = set(dates[train_days : train_days + val_days])
    test_dates = set(dates[train_days + val_days :])
    return train_dates, val_dates, test_dates


def build_Y_from_impulses(df: pd.DataFrame, phi: float) -> np.ndarray:
    Y = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("trade_date").groups.items():
        idx = list(idx)
        y = 0.0
        for i in idx:
            Y[i] = y
            y = phi * y + float(df.at[i, "event_impulse"])
    return Y


def fit_component(train_df: pd.DataFrame, xcol: str, regime_val: int, clock_mean: dict[int, float]) -> dict:
    sub = train_df[train_df["regime"] == regime_val].copy()
    if len(sub) < 25:
        return {
            "component": xcol,
            "regime": regime_val,
            "beta_mean": 0.0,
            "beta_y": 0.0,
            "resid_std": np.nan,
            "n_obs": len(sub),
        }

    x = sub[xcol].to_numpy(dtype=float)
    x_next = sub[f"{xcol}_next"].to_numpy(dtype=float)
    theta = sub["minute_of_day"].map(clock_mean).to_numpy(dtype=float)
    Y = sub["Y"].to_numpy(dtype=float)

    # Reduced-form discrete analogue of:
    # dX_t = kappa (theta_clock(t)-X_t) dt + a Y_t dt + noise
    target = x_next - x
    X = np.column_stack([theta - x, Y])

    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    beta_mean = max(float(beta[0]), 0.0)     # enforce nonnegative mean reversion speed
    beta_y = float(beta[1])

    pred = x + beta_mean * (theta - x) + beta_y * Y
    resid = x_next - pred

    return {
        "component": xcol,
        "regime": regime_val,
        "beta_mean": beta_mean,
        "beta_y": beta_y,
        "resid_std": float(np.std(resid, ddof=1)),
        "n_obs": len(sub),
    }


def fit_reduced_form(train_df: pd.DataFrame) -> tuple[dict, dict]:
    clock_means = {}
    params = {}

    for xcol in ["v_atm", "skew", "curvature_bfly"]:
        clock_means[xcol] = train_df.groupby("minute_of_day")[xcol].mean().to_dict()
        params[xcol] = {}
        for regime_val in [0, 1]:
            params[xcol][regime_val] = fit_component(train_df, xcol, regime_val, clock_means[xcol])

    return clock_means, params


def apply_reduced_form(df: pd.DataFrame, clock_means: dict, params: dict) -> pd.DataFrame:
    out = df.copy()

    for xcol in ["v_atm", "skew", "curvature_bfly"]:
        out[f"{xcol}_pred"] = np.nan
        for regime_val in [0, 1]:
            mask = out["regime"] == regime_val
            theta = out.loc[mask, "minute_of_day"].map(clock_means[xcol]).fillna(out.loc[mask, xcol]).to_numpy(dtype=float)
            x = out.loc[mask, xcol].to_numpy(dtype=float)
            Y = out.loc[mask, "Y"].to_numpy(dtype=float)

            p = params[xcol][regime_val]
            pred = x + p["beta_mean"] * (theta - x) + p["beta_y"] * Y
            out.loc[mask, f"{xcol}_pred"] = pred

    out["v_atm_pred"] = np.maximum(out["v_atm_pred"], 1e-8)
    out["sigma_atm_pred"] = np.sqrt(out["v_atm_pred"])
    out["curvature_bfly_pred"] = out["curvature_bfly_pred"].clip(-CURV_CLIP, CURV_CLIP)
    return out


def feature_prediction_loss(df: pd.DataFrame) -> float:
    loss = 0.0
    for xcol in ["v_atm", "skew", "curvature_bfly"]:
        scale = max(float(df[xcol].std()), 1e-8)
        err = (df[f"{xcol}_next"] - df[f"{xcol}_pred"]) / scale
        loss += float(np.mean(err ** 2))
    return loss


def select_phi(train_trans: pd.DataFrame, val_trans: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []

    for halflife in SHOCK_HALFLIFE_GRID_MIN:
        phi = 2.0 ** (-1.0 / halflife)

        tr = train_trans.copy().reset_index(drop=True)
        va = val_trans.copy().reset_index(drop=True)
        tr["Y"] = build_Y_from_impulses(tr, phi)
        va["Y"] = build_Y_from_impulses(va, phi)

        clock_means, params = fit_reduced_form(tr)
        va_pred = apply_reduced_form(va, clock_means, params)
        loss = feature_prediction_loss(va_pred)

        rows.append({"shock_halflife_min": halflife, "phi_y": phi, "val_feature_loss": loss})

    score_df = pd.DataFrame(rows).sort_values("val_feature_loss").reset_index(drop=True)
    best_phi = float(score_df.loc[0, "phi_y"])
    return best_phi, score_df


def params_to_table(params: dict, phi_y: float) -> pd.DataFrame:
    rows = []

    for xcol in ["v_atm", "skew", "curvature_bfly"]:
        for regime_val in [0, 1]:
            p = params[xcol][regime_val]
            beta_mean = float(p["beta_mean"])
            half_life = np.inf if beta_mean <= 0 else math.log(2.0) / beta_mean
            rows.append(
                {
                    "component": xcol,
                    "regime": regime_val,
                    "regime_name": "fast" if regime_val == 1 else "slow",
                    "mean_reversion_per_min": beta_mean,
                    "mean_reversion_half_life_min": half_life,
                    "shock_loading_beta_y": float(p["beta_y"]),
                    "resid_std": float(p["resid_std"]),
                    "n_obs": int(p["n_obs"]),
                    "phi_y": phi_y,
                }
            )

    return pd.DataFrame(rows)

def main(
    INPUT_DIR: str = str(DEFAULT_INPUT_DIR),
    OUTPUT_DIR: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    in_dir = Path(INPUT_DIR).expanduser()
    out_dir = Path(OUTPUT_DIR).expanduser()

    in_path = in_dir / FEATURES_FILE
    if not in_path.exists():
        raise FileNotFoundError(f"Expected feature file: {in_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading features from {in_path}...")

    feat = load_feature_panel(in_path)
    trans = make_feature_transitions(feat)

    all_dates = sorted(trans["trade_date"].dt.normalize().unique())
    train_dates, val_dates, test_dates = split_dates(all_dates, TRAIN_DAYS, VAL_DAYS)

    train_trans = trans[trans["trade_date"].isin(train_dates)].copy().reset_index(drop=True)
    val_trans = trans[trans["trade_date"].isin(val_dates)].copy().reset_index(drop=True)
    test_trans = trans[trans["trade_date"].isin(test_dates)].copy().reset_index(drop=True)

    print("Selecting optimal shock half-life (phi)...")
    best_phi, phi_scores = select_phi(train_trans, val_trans)

    # Refit on train+val with chosen phi
    print("Fitting reduced-form model on train + validation sets...")
    tv_trans = trans[trans["trade_date"].isin(train_dates | val_dates)].copy().reset_index(drop=True)
    tv_trans["Y"] = build_Y_from_impulses(tv_trans, best_phi)

    clock_means, params = fit_reduced_form(tv_trans)

    # Apply to test set to generate predictions
    print("Generating feature predictions for test set...")
    test_trans["Y"] = build_Y_from_impulses(test_trans.reset_index(drop=True), best_phi)
    test_pred = apply_reduced_form(test_trans, clock_means, params)

    # Export results
    print("Exporting calibration data...")
    param_table = params_to_table(params, best_phi)
    param_table.to_csv(out_dir / "boundary_layer_calibrated_parameters.csv", index=False)
    phi_scores.to_csv(out_dir / "boundary_layer_shock_halflife_grid.csv", index=False)
    test_pred.to_csv(out_dir / "boundary_layer_test_feature_predictions.csv", index=False)

    print("\n========================================")
    print("Model Calibration Complete")
    print("========================================")
    print(f"Chosen shock-state phi_y: {best_phi:.4f}")
    
    print("\nCalibrated Parameters Snapshot:")
    print(param_table[['component', 'regime_name', 'mean_reversion_per_min', 'shock_loading_beta_y']].head())
    
    print("\nPredictions saved. You are ready to run BackTestFinal.py.")


if __name__ == "__main__":
    main()

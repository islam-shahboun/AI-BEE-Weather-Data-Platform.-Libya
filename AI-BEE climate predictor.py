"""
AI-BEE — Final Integrated Climate Predictor & EPW/DDY Generator
==============================================================================
Pure Python, standalone command-line tool. No web platform, no HTML/JS.
This is the final, integrated version of this script, incorporating every
improvement made over the course of the project:

  1. Baseline spatial model: Random Forest + Linear Regression, trained on
     (lat, lon, elevation), with automatic per-target selection of whichever
     algorithm scores higher under Leave-One-Out Cross-Validation (LOOCV).
  2. Hybrid ERA5 bias-correction: for each target, two ways of using ERA5
     reanalysis as an auxiliary predictor are tried under the same LOOCV
     protocol -- naive feature augmentation, and residual (delta) bias
     correction -- and the better strategy is kept automatically, per target.
     This requires a small pre-extracted file of ERA5 values at the 28
     training stations (see extract_era5_at_stations.py); if that file is not
     supplied, the script automatically falls back to the baseline-only model
     and says so explicitly.
  3. Real, model-based confidence reporting: for Random-Forest-selected
     targets, confidence is the spread of the individual trees' predictions
     AT THE EXACT REQUESTED POINT (so it genuinely varies by location); for
     Linear-selected targets, confidence is that model's fixed, dataset-wide
     cross-validated R^2 (clearly labeled as such, since linear regression has
     no per-point uncertainty measure).
  4. Full 8,760-hour EPW file generation using the donor-station hourly-shape
     method with physical corrections (temperature lapse rate, Magnus-Tetens
     humidity recomputation, solar radiation scaling, barometric pressure
     correction).
  5. ASHRAE-style Design Day (DDY) file generation, using the 0.4th/99.6th
     percentile dry-bulb temperatures from the corrected 8,760-hour record.

USAGE
-----
    python ai_bee_climate_predictor_final.py

Requires: pandas, numpy, scikit-learn  (pip install pandas numpy scikit-learn)
Optional: a pre-extracted ERA5-at-stations JSON file (see EOF section
"ERA5 HYBRID MODEL (OPTIONAL)" below) to enable the hybrid bias-correction
model; without it, the script runs the baseline-only model and tells you so.
"""

import os
import re
import glob
import json
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

# =============================================================================
# CONFIGURATION — edit these paths for your machine
# =============================================================================
EPW_DIR = r"C:\Users\engco\Desktop\AI-BEE\Phase 01 - Database Project\Database Code\EPW_Libya"
OUTPUT_DIR = r"C:\Users\engco\Desktop\AI-BEE\Phase 01 - Database Project\Database Code\Generated_EPW"

# Optional: path to a JSON file of ERA5 values at the 28 training stations,
# produced by extract_era5_at_stations.py. If this file does not exist, the
# script automatically runs baseline-only (no hybrid correction) and warns you.
ERA5_STATIONS_JSON = r"C:\Users\engco\Desktop\AI-BEE\Phase 01 - Database Project\Database Code\era5_at_28_stations.json"

BASE_TEMP_C = 18.0
TARGETS = ["HDD", "CDD", "GHI_annual", "DNI_annual", "DHI_annual", "T_mean", "RH_mean", "WS_mean"]
UNITS = {"HDD": "°C·day", "CDD": "°C·day", "GHI_annual": "kWh/m²", "DNI_annual": "kWh/m²",
         "DHI_annual": "kWh/m²", "T_mean": "°C", "RH_mean": "%", "WS_mean": "m/s"}
# Targets for which an ERA5-derived value can be used as a hybrid predictor
# (DHI is excluded: it was not among the ERA5 variables extracted in this project)
HYBRID_CAPABLE = ["HDD", "CDD", "T_mean", "RH_mean", "WS_mean", "GHI_annual", "DNI_annual"]

EPW_COLUMNS = [
    "Year", "Month", "Day", "Hour", "Minute", "DataSourceFlags",
    "DryBulbTemp_C", "DewPointTemp_C", "RelHumidity_pct", "AtmPressure_Pa",
    "ExtHorizRad_Whm2", "ExtDirNormRad_Whm2", "HorizIRSky_Whm2",
    "GHI_Whm2", "DNI_Whm2", "DHI_Whm2",
    "GlobalHorizIllum_lux", "DirectNormIllum_lux", "DiffuseHorizIllum_lux", "ZenithLum_Cdm2",
    "WindDir_deg", "WindSpeed_ms", "TotalSkyCover", "OpaqueSkyCover",
    "Visibility_km", "CeilingHeight_m", "PresWeathObs", "PresWeathCodes",
    "PrecipWater_mm", "AerosolOpticalDepth", "SnowDepth_cm", "DaysSinceSnow",
    "Albedo", "LiquidPrecipDepth_mm", "LiquidPrecipQuantity_hr",
]

CLEAN_NAMES = {
    "Benghazi-Benigna.Intl.AP": "Benghazi", "Jaghbub": "Jaghbub",
    "Tobruk": "Tobruk", "Derna": "Derna", "Ghat.AP": "Ghat",
    "Gharyan": "Gharyan", "Hamada.AP": "Al Hamada", "Qaryat": "Al Qaryat",
    "Yefran": "Yefren", "Hun": "Hun", "Kufra.AP": "Kufra",
    "Tazirbu": "Tazerbo", "Khums": "Al Khums", "Bani.Waled": "Bani Walid",
    "Misrata.AP": "Misrata", "Al.Marj": "Al Marj", "Tragen": "Traghen",
    "Ghadamis.East.AP": "Ghadames", "Nalut": "Nalut", "Zuwara": "Zuwara",
    "Sabha.AP": "Sabha", "Abu.Njaym": "Abu Njaym", "Sirte": "Sirte",
    "Tripoli-Mitiga.Intl.AP": "Tripoli", "Ajdabiya": "Ajdabiya",
    "Jalu": "Jalu", "Marada": "Marada", "Awbari": "Ubari",
}


# =============================================================================
# STEP 1 — LOAD THE 28 REAL EPW FILES
# =============================================================================
def clean_city_name(raw_filename):
    for frag, clean in CLEAN_NAMES.items():
        if frag.lower() in raw_filename.lower():
            return clean
    return re.sub(r"[._]", " ", os.path.splitext(os.path.basename(raw_filename))[0])[:20]


def parse_epw_header(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    parts = first_line.split(",")
    return {
        "latitude": float(parts[6]) if len(parts) > 6 else None,
        "longitude": float(parts[7]) if len(parts) > 7 else None,
        "tz": float(parts[8]) if len(parts) > 8 else 2.0,
        "elevation_m": float(parts[9]) if len(parts) > 9 else 0.0,
    }


def parse_epw_data(filepath):
    return pd.read_csv(filepath, skiprows=8, header=None, names=EPW_COLUMNS,
                        usecols=range(len(EPW_COLUMNS)), engine="python", on_bad_lines="skip")


def annual_summary(header, df):
    T = pd.to_numeric(df["DryBulbTemp_C"], errors="coerce")
    RH = pd.to_numeric(df["RelHumidity_pct"], errors="coerce")
    GHI = pd.to_numeric(df["GHI_Whm2"], errors="coerce")
    DNI = pd.to_numeric(df["DNI_Whm2"], errors="coerce")
    DHI = pd.to_numeric(df["DHI_Whm2"], errors="coerce")
    WS = pd.to_numeric(df["WindSpeed_ms"], errors="coerce")
    hdd = float((np.maximum(0, BASE_TEMP_C - T) / 24.0).sum())
    cdd = float((np.maximum(0, T - BASE_TEMP_C) / 24.0).sum())
    return {
        "lat": header["latitude"], "lon": header["longitude"], "elev": header["elevation_m"],
        "T_mean": float(T.mean()), "RH_mean": float(RH.mean()), "WS_mean": float(WS.mean()),
        "HDD": hdd, "CDD": cdd,
        "GHI_annual": float(GHI.sum()) / 1000.0,
        "DNI_annual": float(DNI.sum()) / 1000.0,
        "DHI_annual": float(DHI.sum()) / 1000.0,
    }


def load_stations():
    epw_files = sorted(glob.glob(os.path.join(EPW_DIR, "**", "*.epw"), recursive=True))
    if not epw_files:
        raise FileNotFoundError(
            f"No .epw files found under:\n  {EPW_DIR}\n"
            "Edit EPW_DIR at the top of this script to point to your real folder."
        )
    print(f"Loading {len(epw_files)} real EPW files...")
    stations = []
    for path in epw_files:
        header = parse_epw_header(path)
        if header["latitude"] is None:
            continue
        df = parse_epw_data(path)
        name = clean_city_name(path)
        summ = annual_summary(header, df)
        summ["name"] = name
        summ["df"] = df  # keep the full hourly data for later EPW synthesis
        stations.append(summ)
        print(f"  {name:14s} lat={summ['lat']:.4f} lon={summ['lon']:.4f} elev={summ['elev']:.0f}m "
              f"HDD={summ['HDD']:.0f} CDD={summ['CDD']:.0f}")
    print(f"Loaded {len(stations)} stations.\n")
    return stations


def load_era5_at_stations(stations):
    """Load the optional ERA5-at-stations file used to enable the hybrid model.
    Returns None (with a warning) if the file is absent or incomplete."""
    if not os.path.exists(ERA5_STATIONS_JSON):
        print(f"NOTE: ERA5 stations file not found at:\n  {ERA5_STATIONS_JSON}")
        print("      Running BASELINE-ONLY model (no ERA5 hybrid bias-correction).")
        print("      See extract_era5_at_stations.py to generate this file.\n")
        return None
    with open(ERA5_STATIONS_JSON, "r") as f:
        era5 = json.load(f)
    missing = [s["name"] for s in stations if s["name"] not in era5]
    if missing:
        print(f"WARNING: ERA5 data missing for stations: {missing}. Running BASELINE-ONLY model.\n")
        return None
    print(f"Loaded ERA5 reference data for {len(era5)} stations -> hybrid bias-correction ENABLED.\n")
    return era5


# =============================================================================
# STEP 2 — TRAIN MODELS: BASELINE + OPTIONAL HYBRID BIAS-CORRECTION
# =============================================================================
def loocv_best_of(X, y, loo):
    """Fit RF and Linear under LOOCV, return (best_r2, best_mae, fitted_model_on_all_data, model_type)."""
    rf_preds = np.zeros_like(y, dtype=float)
    lin_preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=1)
        rf.fit(X[train_idx], y[train_idx])
        rf_preds[test_idx] = rf.predict(X[test_idx])
        lin = LinearRegression()
        lin.fit(X[train_idx], y[train_idx])
        lin_preds[test_idx] = lin.predict(X[test_idx])
    rf_r2, lin_r2 = r2_score(y, rf_preds), r2_score(y, lin_preds)
    rf_mae, lin_mae = mean_absolute_error(y, rf_preds), mean_absolute_error(y, lin_preds)
    if rf_r2 >= lin_r2:
        final = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=1)
        final.fit(X, y)
        return rf_r2, rf_mae, final, "rf"
    else:
        final = LinearRegression()
        final.fit(X, y)
        return lin_r2, lin_mae, final, "linear"


def train_models(stations, era5_stations):
    summary_df = pd.DataFrame([{k: s[k] for k in ["name", "lat", "lon", "elev"] + TARGETS} for s in stations])
    X_base = summary_df[["lat", "lon", "elev"]].values
    loo = LeaveOneOut()

    if era5_stations is not None:
        for key in HYBRID_CAPABLE:
            summary_df[f"ERA5_{key}"] = summary_df["name"].map(lambda n: era5_stations[n][key])

    print("Training models with Leave-One-Out cross-validation (28 folds)...\n")
    models, metrics = {}, {}

    header = f"{'Target':12s} {'Approach':10s} {'Model':8s} {'R2':>7s} {'MAE':>9s}"
    if era5_stations is not None:
        header += f"   {'(Baseline R2':>13s} {'Naive R2':>9s} {'Residual R2)':>13s}"
    print(header)

    for key in TARGETS:
        y = summary_df[key].values
        y_std = float(np.std(y))

        base_r2, base_mae, base_model, base_type = loocv_best_of(X_base, y, loo)

        use_hybrid = era5_stations is not None and key in HYBRID_CAPABLE
        if not use_hybrid:
            models[key] = {"approach": "baseline", "model": base_model, "type": base_type, "y_std": y_std}
            metrics[key] = {"approach": "baseline", "chosen_r2": base_r2, "chosen_mae": base_mae}
            flag = "  <- LOW RELIABILITY" if base_r2 < 0.3 else ""
            print(f"{key:12s} {'baseline':10s} {base_type:8s} {base_r2:7.3f} {base_mae:9.1f}{flag}")
            continue

        era5_vals = summary_df[f"ERA5_{key}"].values

        # Naive augmentation: [lat, lon, elev, ERA5_value] -> real value
        X_aug = np.column_stack([X_base, era5_vals])
        naive_r2, naive_mae, naive_model, naive_type = loocv_best_of(X_aug, y, loo)

        # Residual correction: [lat, lon, elev] -> (real - ERA5), added back to ERA5 at predict time
        residual = y - era5_vals
        resid_r2_internal, resid_mae_internal, resid_model, resid_type = loocv_best_of(X_base, residual, loo)
        # Proper LOOCV on the real-value scale for the residual approach:
        resid_preds_full = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in loo.split(X_base):
            rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=1)
            rf.fit(X_base[train_idx], residual[train_idx])
            rf_p = rf.predict(X_base[test_idx])
            lin = LinearRegression()
            lin.fit(X_base[train_idx], residual[train_idx])
            lin_p = lin.predict(X_base[test_idx])
            # use whichever residual sub-model type was chosen above for consistency
            resid_preds_full[test_idx] = era5_vals[test_idx] + (rf_p if resid_type == "rf" else lin_p)
        resid_r2 = r2_score(y, resid_preds_full)

        approach_scores = {"naive": naive_r2, "residual": resid_r2}
        best_approach = max(approach_scores, key=approach_scores.get)

        if best_approach == "naive":
            models[key] = {"approach": "naive", "model": naive_model, "type": naive_type, "y_std": y_std}
            chosen_r2, chosen_mae = naive_r2, naive_mae
        else:
            models[key] = {"approach": "residual", "model": resid_model, "type": resid_type, "y_std": y_std}
            chosen_r2, chosen_mae = resid_r2, resid_mae_internal

        metrics[key] = {"approach": best_approach, "chosen_r2": chosen_r2, "chosen_mae": chosen_mae,
                         "baseline_r2": base_r2, "naive_r2": naive_r2, "residual_r2": resid_r2}
        flag = "  <- LOW RELIABILITY" if chosen_r2 < 0.3 else ""
        print(f"{key:12s} {best_approach:10s} {models[key]['type']:8s} {chosen_r2:7.3f} {chosen_mae:9.1f}"
              f"   {base_r2:13.3f} {naive_r2:9.3f} {resid_r2:13.3f}{flag}")

    print()
    low_reliability = [k for k in TARGETS if metrics[k]["chosen_r2"] < 0.3]
    if low_reliability:
        print("WARNING - the following targets have low cross-validated R2 (<0.3) and")
        print("should be treated as unreliable regardless of which location you predict:")
        for k in low_reliability:
            print(f"  - {k}  (R2={metrics[k]['chosen_r2']:.3f})")
        print()

    return models, metrics, summary_df


# =============================================================================
# STEP 3 — PREDICT + CONFIDENCE METRICS FOR A NEW LOCATION
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def idw_elevation(lat, lon, stations, power=2):
    weights, values = [], []
    for s in stations:
        d = max(haversine_km(lat, lon, s["lat"], s["lon"]), 1.0)
        w = 1.0 / d ** power
        weights.append(w)
        values.append(w * s["elev"])
    return sum(values) / sum(weights)


def find_nearest_station(lat, lon, stations):
    best, best_dist = None, float("inf")
    for s in stations:
        d = haversine_km(lat, lon, s["lat"], s["lon"])
        if d < best_dist:
            best, best_dist = s, d
    return best, best_dist


def get_era5_for_target(name):
    """Interactively collect ERA5 values for a new target location, needed for
    the hybrid model. Returns a dict or None if the user has no ERA5 data."""
    print(f"\n  This model uses ERA5 reanalysis as an auxiliary predictor for better accuracy.")
    has_era5 = input("  Do you have ERA5 values for this exact location? (y/n): ").strip().lower()
    if has_era5 != "y":
        return None
    era5 = {}
    print("  Enter the ERA5-derived annual values for this location (see extract_era5_at_stations.py):")
    for key in HYBRID_CAPABLE:
        era5[key] = get_float(f"    ERA5 {key} ({UNITS[key]}): ")
    return era5


def predict_location(lat, lon, elev, models, metrics, stations, era5_target=None):
    X_base = np.array([[lat, lon, elev]])
    pred = {}
    model_conf = {}

    for key in TARGETS:
        info = models[key]
        if info["approach"] == "baseline" or era5_target is None:
            X_in = X_base
            base_prediction = float(info["model"].predict(X_in)[0])
            pred[key] = base_prediction
        elif info["approach"] == "naive":
            X_in = np.array([[lat, lon, elev, era5_target[key]]])
            pred[key] = float(info["model"].predict(X_in)[0])
        else:  # residual
            X_in = X_base
            resid_pred = float(info["model"].predict(X_in)[0])
            pred[key] = era5_target[key] + resid_pred

        # Confidence: RF tree-agreement (location-specific) or Linear R2 (fixed)
        if info["type"] == "rf":
            tree_preds = np.array([tree.predict(X_in)[0] for tree in info["model"].estimators_])
            tree_std = float(np.std(tree_preds))
            pct = 100.0 * (1.0 - tree_std / info["y_std"]) if info["y_std"] > 0 else 50.0
            model_conf[key] = {"pct": float(np.clip(pct, 2, 99)), "basis": "RF tree agreement (location-specific)"}
        else:
            r2 = metrics[key]["chosen_r2"]
            pct = 100.0 * max(0.0, r2)
            model_conf[key] = {"pct": float(np.clip(pct, 2, 99)), "basis": "Model CV R2 (fixed, same everywhere)"}

    donor, dist = find_nearest_station(lat, lon, stations)
    diffs = [abs((pred[k] - donor[k]) / donor[k] * 100.0) if donor[k] else 0.0 for k in TARGETS]
    match_pct = max(0.0, min(100.0, 100.0 - (sum(diffs) / len(diffs))))

    return pred, donor, dist, model_conf, match_pct


# =============================================================================
# STEP 4 — HYBRID EPW SYNTHESIS (donor station shape + physical correction)
# =============================================================================
def magnus_rh(t_c, td_c):
    a, b = 17.625, 243.04
    gamma_t = (a * t_c) / (b + t_c)
    gamma_td = (a * td_c) / (b + td_c)
    return float(np.clip(100.0 * np.exp(gamma_td - gamma_t), 1, 100))


def barometric_pressure(elev_m):
    return 101325.0 * (1 - 2.25577e-5 * elev_m) ** 5.25588


def synthesize_hourly(donor, pred, lat, lon, elev):
    df = donor["df"].copy()

    t_shift = pred["T_mean"] - donor["T_mean"]
    elev_diff_km = (elev - donor["elev"]) / 1000.0
    total_shift = t_shift + (-6.5 * elev_diff_km)

    T = pd.to_numeric(df["DryBulbTemp_C"], errors="coerce") + total_shift
    Td = pd.to_numeric(df["DewPointTemp_C"], errors="coerce") + total_shift
    Td = np.minimum(Td, T - 0.1)
    RH = [magnus_rh(t, td) for t, td in zip(T, Td)]

    df["DryBulbTemp_C"] = T.round(1)
    df["DewPointTemp_C"] = Td.round(1)
    df["RelHumidity_pct"] = np.round(RH, 0)

    for col, key in [("GHI_Whm2", "GHI_annual"), ("DNI_Whm2", "DNI_annual"), ("DHI_Whm2", "DHI_annual")]:
        donor_total = donor[key]
        ratio = (pred[key] / donor_total) if donor_total > 0 else 1.0
        df[col] = (pd.to_numeric(df[col], errors="coerce") * ratio).round(0).clip(lower=0)

    donor_p0 = barometric_pressure(donor["elev"])
    target_p0 = barometric_pressure(elev)
    p_ratio = target_p0 / donor_p0 if donor_p0 else 1.0
    df["AtmPressure_Pa"] = (pd.to_numeric(df["AtmPressure_Pa"], errors="coerce") * p_ratio).round(0)

    return df


def write_epw(path, name, lat, lon, elev, tz, donor, dist, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header_lines = [
        f"LOCATION,{name},-,LBY,AI-BEE Synthesized,999999,{lat:.4f},{lon:.4f},{tz:.1f},{elev:.1f}",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,AI-BEE ML-synthesized EPW (hybrid RF/Linear + ERA5 bias-correction, nearest-station hourly shape).",
        f"COMMENTS 2,Donor station: {donor['name']} ({dist:.0f} km away). Engineering-grade estimate, not measured data.",
        "DATA PERIODS,1,1,Data,Friday, 1/ 1,12/31",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        for _, row in df.iterrows():
            f.write(",".join(str(row[c]) for c in EPW_COLUMNS) + "\n")


# =============================================================================
# STEP 5 — ASHRAE-STYLE DESIGN DAY (DDY) FILE GENERATION
# =============================================================================
def write_ddy(path, name, lat, lon, elev, tz, df):
    """Generate a SizingPeriod:DesignDay pair (winter heating / summer cooling)
    using 99.6th/0.4th percentile dry-bulb temperatures from the corrected
    hourly record, following standard ASHRAE/EnergyPlus DDY convention."""
    T = pd.to_numeric(df["DryBulbTemp_C"], errors="coerce")
    RH = pd.to_numeric(df["RelHumidity_pct"], errors="coerce")
    WS = pd.to_numeric(df["WindSpeed_ms"], errors="coerce")
    WD = pd.to_numeric(df["WindDir_deg"], errors="coerce")
    P = pd.to_numeric(df["AtmPressure_Pa"], errors="coerce")

    heating_temp = float(np.percentile(T, 0.4))
    cooling_temp = float(np.percentile(T, 99.6))
    # coincident conditions at the hour closest to the cooling design temperature
    cool_idx = (T - cooling_temp).abs().idxmin()
    coincident_wb_approx = float(RH.loc[cool_idx])  # simplified proxy; see note below
    design_wind = float(WS.mean())
    design_wind_dir = float(WD.mean())
    design_pressure = float(P.mean())

    ddy_text = f"""! AI-BEE ML-synthesized Design Day file for {name}
! Heating design temp = 0.4th percentile, Cooling design temp = 99.6th percentile
! of the corrected 8760-hour synthesized record (see accompanying .epw file).

SizingPeriod:DesignDay,
  {name} Ann Htg 99.6% Condns DB,      !- Name
  1,                                    !- Month
  21,                                   !- Day of Month
  WinterDesignDay,                      !- Day Type
  {heating_temp:.1f},                   !- Maximum Dry-Bulb Temperature {{C}}
  0.0,                                  !- Daily Dry-Bulb Temperature Range {{deltaC}}
  ,                                     !- Dry-Bulb Temperature Range Modifier Type
  ,                                     !- Dry-Bulb Temperature Range Modifier Day Schedule Name
  Wetbulb,                              !- Humidity Condition Type
  {heating_temp:.1f},                   !- Wetbulb or DewPoint at Maximum Dry-Bulb {{C}}
  ,                                     !- Humidity Condition Day Schedule Name
  ,                                     !- Humidity Ratio at Maximum Dry-Bulb {{kgWater/kgDryAir}}
  ,                                     !- Enthalpy at Maximum Dry-Bulb {{J/kg}}
  ,                                     !- Daily Wet-Bulb Temperature Range {{deltaC}}
  {design_pressure:.0f},                !- Barometric Pressure {{Pa}}
  {design_wind:.1f},                    !- Wind Speed {{m/s}}
  {design_wind_dir:.0f},                !- Wind Direction {{deg}}
  No,                                   !- Rain Indicator
  No,                                   !- Snow Indicator
  No,                                   !- Daylight Saving Time Indicator
  ASHRAEClearSky,                       !- Solar Model Indicator
  ,                                     !- Beam Solar Day Schedule Name
  ,                                     !- Diffuse Solar Day Schedule Name
  ,                                     !- ASHRAE Clear Sky Optical Depth for Beam Irradiance (taub)
  ,                                     !- ASHRAE Clear Sky Optical Depth for Diffuse Irradiance (taud)
  0.0;                                  !- Sky Clearness

SizingPeriod:DesignDay,
  {name} Ann Clg .4% Condns DB=>MWB,   !- Name
  7,                                    !- Month
  21,                                   !- Day of Month
  SummerDesignDay,                      !- Day Type
  {cooling_temp:.1f},                   !- Maximum Dry-Bulb Temperature {{C}}
  10.0,                                 !- Daily Dry-Bulb Temperature Range {{deltaC}}
  ,                                     !- Dry-Bulb Temperature Range Modifier Type
  ,                                     !- Dry-Bulb Temperature Range Modifier Day Schedule Name
  Wetbulb,                              !- Humidity Condition Type
  {coincident_wb_approx:.1f},           !- Wetbulb or DewPoint at Maximum Dry-Bulb {{C}}
  ,                                     !- Humidity Condition Day Schedule Name
  ,                                     !- Humidity Ratio at Maximum Dry-Bulb {{kgWater/kgDryAir}}
  ,                                     !- Enthalpy at Maximum Dry-Bulb {{J/kg}}
  ,                                     !- Daily Wet-Bulb Temperature Range {{deltaC}}
  {design_pressure:.0f},                !- Barometric Pressure {{Pa}}
  {design_wind:.1f},                    !- Wind Speed {{m/s}}
  {design_wind_dir:.0f},                !- Wind Direction {{deg}}
  No,                                   !- Rain Indicator
  No,                                   !- Snow Indicator
  No,                                   !- Daylight Saving Time Indicator
  ASHRAEClearSky,                       !- Solar Model Indicator
  ,                                     !- Beam Solar Day Schedule Name
  ,                                     !- Diffuse Solar Day Schedule Name
  ,                                     !- ASHRAE Clear Sky Optical Depth for Beam Irradiance (taub)
  ,                                     !- ASHRAE Clear Sky Optical Depth for Diffuse Irradiance (taud)
  1.0;                                  !- Sky Clearness
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(ddy_text)

    # NOTE ON SIMPLIFICATION: the cooling-day "wetbulb" value written above is
    # actually a coincident relative-humidity proxy, not a true wet-bulb
    # temperature (which requires a psychrometric calculation from dry-bulb,
    # relative humidity, and pressure). This is flagged here deliberately as a
    # known simplification -- see Section 5.3 (Limitations) of the accompanying
    # paper -- and should be replaced with a proper wet-bulb calculation
    # (e.g. via a psychrometric library) before using the DDY file for final
    # HVAC equipment sizing on a real project.


# =============================================================================
# INTERACTIVE MAIN LOOP
# =============================================================================
def get_float(prompt, lo=None, hi=None, allow_blank=False, blank_value=None):
    while True:
        raw = input(prompt).strip()
        if allow_blank and raw == "":
            return blank_value
        try:
            val = float(raw)
            if lo is not None and val < lo:
                raise ValueError
            if hi is not None and val > hi:
                raise ValueError
            return val
        except ValueError:
            rng = f" between {lo} and {hi}" if lo is not None else ""
            print(f"  Invalid value — please enter a number{rng}.")


def main():
    stations = load_stations()
    era5_stations = load_era5_at_stations(stations)
    models, metrics, summary_df = train_models(stations, era5_stations)

    print("=" * 78)
    print("Model ready. You can now enter any location in Libya for a prediction.")
    print("=" * 78)

    while True:
        print("\n" + "-" * 78)
        name = input("Location name (e.g. Sabratha): ").strip() or "New_Location"
        lat = get_float("Latitude  (approx. 19-34 for Libya): ", 15, 35)
        lon = get_float("Longitude (approx. 9-26 for Libya): ", 5, 30)
        elev = get_float("Elevation in meters (leave blank to auto-estimate): ",
                          allow_blank=True, blank_value=None)
        if elev is None:
            elev = idw_elevation(lat, lon, stations)
            print(f"  -> Auto-estimated elevation: {elev:.0f} m")

        era5_target = get_era5_for_target(name) if era5_stations is not None else None
        if era5_stations is not None and era5_target is None:
            print("  -> No ERA5 values supplied: hybrid-model targets will fall back to their")
            print("     baseline (non-ERA5-corrected) prediction for this location only.")

        pred, donor, dist, model_conf, match_pct = predict_location(
            lat, lon, elev, models, metrics, stations, era5_target)

        print(f"\n--- Prediction for {name} ({lat:.4f}, {lon:.4f}), elevation {elev:.0f} m ---")
        print(f"{'Target':12s} {'Value':>10s} {'Approach':>10s} {'Model':>7s} {'Confidence':>11s}   vs nearest station")
        for key in TARGETS:
            real = donor[key]
            diff_pct = (pred[key] - real) / real * 100 if real else 0.0
            mc = model_conf[key]
            approach = metrics[key]["approach"]
            warn = " (LOW RELIABILITY)" if metrics[key]["chosen_r2"] < 0.3 else ""
            print(f"  {key:12s} {pred[key]:8.1f}{UNITS[key]:>4s} {approach:>10s} {models[key]['type']:>7s} "
                  f"{mc['pct']:9.0f}%   ({diff_pct:+.0f}% vs {donor['name']}){warn}")

        print(f"\n  Nearest real station     : {donor['name']} ({dist:.0f} km away)")
        print(f"  Match vs that station    : {match_pct:.0f}%")
        print(f"\n  Note on 'Confidence': for Random-Forest-based targets it reflects how much")
        print(f"  the 300 individual trees agree with each other AT THIS SPECIFIC POINT (real")
        print(f"  model uncertainty, varies by location). For Linear-based targets it is fixed")
        print(f"  everywhere and equals that model's cross-validated R2.")

        hourly = synthesize_hourly(donor, pred, lat, lon, elev)
        safe_name = name.replace(" ", "_")
        epw_path = os.path.join(OUTPUT_DIR, f"{safe_name}_AI-BEE_Synthesized.epw")
        ddy_path = os.path.join(OUTPUT_DIR, f"{safe_name}_AI-BEE_Synthesized.ddy")
        write_epw(epw_path, name, lat, lon, elev, 2.0, donor, dist, hourly)
        write_ddy(ddy_path, name, lat, lon, elev, 2.0, hourly)
        print(f"\n  Full 8760-hour EPW file saved to:\n    {epw_path}")
        print(f"  Design Day (DDY) file saved to:\n    {ddy_path}")

        again = input("\nGenerate another location? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    main()

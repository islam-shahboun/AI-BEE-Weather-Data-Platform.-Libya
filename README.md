# AI-BEE Weather Data Platform

**A hybrid machine-learning and reanalysis-corrected climate prediction platform for Libya**, built on 28 real TMYx weather stations (2011–2025) and ERA5 reanalysis. Predicts climate metrics (HDD, CDD, solar radiation, temperature, humidity, wind) for **any location in Libya** and generates ready-to-use EnergyPlus `.epw` weather files and ASHRAE-style `.ddy` design-day files — no measured station required at the target site.

**Created by:** Eng. Islam K Shahboun & Prof. Dr. Samah K Alghoul

---

## Live Demo

Open `platform/AI-BEE_Weather_Platform.html` directly in any browser — no installation, no server needed. All computation runs client-side in JavaScript.

## Features

- Interactive map with the 28 real reference stations (exact GPS coordinates extracted from real EPW headers)
- Click anywhere in Libya, or enter coordinates manually, for an instant hybrid ML + ERA5-corrected climate prediction
- **Hybrid bias-correction model**: for each climate variable, automatically selects the best-performing approach (baseline geography-only, ERA5 feature augmentation, or ERA5 residual correction), validated via Leave-One-Out Cross-Validation
- Location-specific confidence metrics based on spatial distance decay and agreement with the nearest real station
- Full 8,760-hour synthetic `.epw` file generation for any point, using a physically-corrected hybrid method (temperature lapse rate, Magnus-Tetens humidity, solar radiation scaling, barometric pressure correction)
- **ASHRAE-style `.ddy` design-day file generation** (0.4th/99.6th percentile heating/cooling design conditions)
- Downloadable Excel export of predictions and the full 28-station reference dataset
- Statistical charts: station ranking, elevation vs. cooling degree-days, climate zone distribution

## Repository Structure

```
├── platform/
│   └── AI-BEE Weather Data Platform.html          # Self-contained interactive platform (hybrid model, EPW+DDY generation)
├── scripts/
│   ├── compute monthly profiles.py           # Compresses hourly data into monthly-hourly profiles for the platform
│   ├── extract era5 at stations.py           # Extracts ERA5 reanalysis values at the 28 training stations
│   ├── extract era5 at validation points.py  # Extracts ERA5 values at independent validation points
│   └── AI-BEE climate predictor.py           # Standalone interactive CLI tool: hybrid model + EPW/DDY generation 
├── docs/
│   ├── AI-BEE_Validation_Report.docx         # Full external validation report (baseline vs. hybrid, statistical tests)
│   └── Table1_Station_Summary.docx           # Complete 28-station coordinate/climate summary table
├── LICENSE
├── CITATION.cff
└── README.md
```

## Quick Start

### Option A — Just use the platform
Download `platform/AI-BEE Weather Data Platform.html` and open it in Chrome/Edge/Firefox. That's it.

### Option B — Run the standalone Python predictor (hybrid model + EPW/DDY generation)
```bash
pip install pandas numpy scikit-learn
python scripts/AI-BEE climate predictor.py
```
Follow the interactive prompts (location name, latitude, longitude, elevation, and optionally ERA5 values for the hybrid correction). It trains the models, reports cross-validated accuracy, and generates full `.epw` and `.ddy` files for your location. If ERA5 data is not supplied, it automatically falls back to the baseline model and tells you so.

### Option C — Regenerate all underlying data from scratch
```bash
python scripts/compute monthly profiles.py              # -> monthly_profiles.json
python scripts/extract era5 at stations.py              # -> era5_at_28_stations.json (enables hybrid bias-correction)
python scripts/extract era5 at validation points.py     # -> era5_at_9_validation_points.json (for independent validation)
```
Edit the path variables at the top of each script to point to your local folders.

## Methodology (summary)

1. **Baseline spatial prediction:** Random Forest Regression (with Linear Regression as an automatically-selected alternative per target) trained on (latitude, longitude, elevation) → annual climate targets, validated with Leave-One-Out Cross-Validation.
2. **Hybrid ERA5 bias-correction:** for each target, compares a naive feature-augmentation strategy against a residual (delta) correction strategy, both using ERA5 reanalysis as an auxiliary predictor; the better-performing strategy per target (via LOOCV) is kept.
3. **Independent external validation:** nine points never used in training — six long-term ground stations (27–53 years of daily records) and three ERA5-derived points at the largest spatial gaps in the training network (up to 451 km from the nearest station) — used to detect and correct a statistically significant systematic bias in heating/cooling degree-days (see `docs/AI-BEE_Validation_Report.docx` for full results).
4. **Hybrid EPW/DDY synthesis:** the nearest real station's hourly *shape* is borrowed, then physically corrected — temperature lapse rate (−6.5 °C/km), solar radiation scaling, barometric pressure correction, relative humidity recomputed via the Magnus-Tetens formula, and ASHRAE-style design-day percentiles for the `.ddy` file.

## Key Validation Result

Independent external validation (n=9) found a statistically significant systematic bias in the baseline model's HDD estimation (paired t-test, p = 0.0012). After hybrid ERA5 bias-correction, this bias was no longer statistically detectable (p = 0.1026), with NRMSE reduced by 23% and R² improved from 0.463 to 0.680. Full statistics for all validated variables are in `docs/AI-BEE_Validation_Report.docx`.

## Known Limitations 

- Trained on only 28 stations — a small sample for any ML model, regardless of cross-validation rigor.
- The ERA5-only validation subset (n=3) is too small for standalone statistical conclusions; treat as illustrative alongside the ground-station results.
- Wind speed prediction (baseline R² ≈ 0.1) remains unreliable in both the baseline and hybrid models and should not be used for design decisions.
- Cooling degree-day and mean temperature bias were substantially reduced by the hybrid correction but not fully eliminated (remain statistically significant at p < 0.05).
- The "donor station" selection for EPW/DDY synthesis is currently nearest-neighbor only (not yet a distance-weighted blend of multiple stations).
- The DDY file's cooling-day "wetbulb" value is a simplified coincident relative-humidity proxy, not a full psychrometric wet-bulb calculation — verify before use in final HVAC equipment sizing.
- Synthesized EPW/DDY files are an engineering-grade estimate for early-stage screening, not a substitute for measured data or a physically-based downscaling tool for final design-stage simulations.

## Data Sources & Citation

The 28 real EPW/DDY files originate from the TMYx dataset:

> Lawrie, Linda K, Drury B Crawley. 2022. *Development of Global Typical Meteorological Years (TMYx)*. https://climate.onebuilding.org

ERA5 reanalysis data: Copernicus Climate Change Service (C3S), Climate Data Store.

Solar position calculations use pvlib python (Holmgren et al., 2018; Anderson et al., 2023), implementing the NREL Solar Position Algorithm (Reda & Andreas, 2004).

Please download the original EPW files directly from climate.onebuilding.org; this repository does not redistribute the raw weather files, only the derived code, models, and platform built from them.

## License

Code in this repository is released under the MIT License (see `LICENSE`). See `CITATION.cff` for how to cite this software.

## Acknowledgements

Part of the AI-BEE (AI-Building Energy Efficiency) project.

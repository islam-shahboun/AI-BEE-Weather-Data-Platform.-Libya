"""
Reduce the 40MB full-hourly dataset to compact monthly x hourly average
profiles per station (12 months x 24 hours), for embedding client-side in
the web platform to synthesize full 8760-hour EPW files without shipping
tens of MB of raw hourly data to the browser.
"""
import json, os
import numpy as np

IN_PATH = r"C:\Users\engco\Desktop\AI-BEE\Phase 01 - Database Project\Database Code\platform_data\stations_hourly.json"
OUT_PATH = r"C:\Users\engco\Desktop\AI-BEE\Phase 01 - Database Project\Database Code\platform_data\monthly_profiles.json"

VARS = ["tdb","tdp","rh","pressure","ghi","dni","dhi","extrad","extdirrad","horizir",
        "windspeed","winddir","totskycvr","opqskycvr","visibility","ceilhgt",
        "precipwtr","aod","snowdepth","dayssincesnow","albedo","liqprecipdepth","liqprecipqty",
        "globillum","dirillum","diffillum","zenlum"]

print("Loading full hourly dataset...")
with open(IN_PATH) as f:
    hourly = json.load(f)

profiles = {}
for station, data in hourly.items():
    month = np.array(data["month"])
    hour = np.array(data["hour"])
    station_profile = {}
    for var in VARS:
        arr = np.array(data[var], dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        matrix = np.zeros((12, 24))
        for m in range(1, 13):
            for h in range(1, 25):
                mask = (month == m) & (hour == h)
                matrix[m - 1, h - 1] = round(float(arr[mask].mean()), 2) if mask.any() else 0.0
        station_profile[var] = matrix.round(2).tolist()
    profiles[station] = station_profile
    print(f"  processed {station}")

with open(OUT_PATH, "w") as f:
    json.dump(profiles, f, separators=(",", ":"))

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"\nSaved monthly-hourly profiles for {len(profiles)} stations -> {OUT_PATH} ({size_kb:.0f} KB)")

import pandas as pd
import numpy as np
import random

random.seed(42)

# ── 1. BRIDGES ──────────────────────────────────────────────
bridge_raw = pd.read_csv("bridge_dataset.csv")

bridge_names = [
    "Adyar Bridge", "Kotturpuram Bridge", "Pammal Bridge",
    "Ambattur Bridge", "Porur Bridge", "Velachery Bridge",
    "Tambaram Bridge", "Mogappair Bridge", "Sholinganallur Bridge",
    "Chromepet Bridge"
]
bridge_coords = [
    (13.0067, 80.2206), (13.0150, 80.2350), (12.9736, 80.1444),
    (13.1143, 80.1548), (13.0389, 80.1567), (12.9815, 80.2209),
    (12.9229, 80.1275), (13.0827, 80.1707), (12.9010, 80.2279),
    (12.9516, 80.1462)
]

bridges = []
total = len(bridge_raw)
for i in range(10):
    row = bridge_raw.iloc[i * (total // 10)]
    lat, lng = bridge_coords[i]
    vibration = round(abs(float(row["acceleration_x"])) * 10, 2)
    temp = round(float(row["temperature_c"]), 2)
    wind = round(float(row["wind_speed_mps"]), 2)
    bridges.append({
        "asset_id": f"BR_{str(i+1).zfill(3)}",
        "name": bridge_names[i],
        "type": "bridge",
        "lat": lat, "lng": lng,
        "vibration": vibration,
        "crack_index": round(random.uniform(0.5, 9.5), 2),
        "load_tons": round(random.uniform(40, 110), 2),
        "temperature": temp,
        "wind_speed": wind,
        "flicker_count": None, "hours_offline": None,
        "tilt_angle": None, "co2": None, "humidity": None,
        "strain": None, "pressure": None, "flow_rate": None,
        "leakage_flag": None, "defect_count": None, "avg_confidence": None
    })

# ── 2. STREETLIGHTS ─────────────────────────────────────────
light_raw = pd.read_csv("smart_lighting_dataset_2024.csv")

light_names = [
    "Anna Nagar Light", "Adyar Signal", "T Nagar Main",
    "Velachery Main", "OMR Junction", "Tambaram Light",
    "Porur Junction", "Mogappair Light", "Guindy Signal",
    "Chromepet Light", "Perambur Signal", "Royapettah Light",
    "Mylapore Signal", "Sholinganallur Light", "Ambattur Signal"
]
light_coords = [
    (13.0827, 80.2101), (13.0067, 80.2206), (13.0418, 80.2341),
    (12.9815, 80.2209), (12.9010, 80.2279), (12.9229, 80.1275),
    (13.0389, 80.1567), (13.0827, 80.1707), (13.0067, 80.2206),
    (12.9516, 80.1462), (13.1167, 80.2333), (13.0543, 80.2627),
    (13.0339, 80.2686), (12.9010, 80.2279), (13.1143, 80.1548)
]

zones = light_raw["zone_id"].unique()
streetlights = []
for i in range(15):
    zone = zones[i % len(zones)]
    subset = light_raw[light_raw["zone_id"] == zone]
    row = subset.iloc[i % len(subset)]
    lat, lng = light_coords[i]
    streetlights.append({
        "asset_id": f"SL_{str(i+1).zfill(3)}",
        "name": light_names[i],
        "type": "streetlight",
        "lat": lat, "lng": lng,
        "vibration": None, "crack_index": None, "load_tons": None,
        "temperature": round(float(row["temperature_celsius"]), 2),
        "wind_speed": None,
        "flicker_count": random.randint(0, 6),
        "hours_offline": round(random.uniform(0, 55), 1),
        "tilt_angle": None, "co2": None, "humidity": None,
        "strain": None, "pressure": None, "flow_rate": None,
        "leakage_flag": None, "defect_count": None, "avg_confidence": None
    })

# ── 3. BUILDINGS ─────────────────────────────────────────────
building_raw = pd.read_csv("building_health_monitoring_dataset.csv")

building_names = [
    "T Nagar Block", "Anna Nagar Tower", "Royapettah Complex",
    "Mylapore Hall", "Guindy Office", "Velachery Complex",
    "OMR Tech Park", "Perambur Block", "Chromepet Hall",
    "Porur Complex"
]
building_coords = [
    (13.0418, 80.2341), (13.0850, 80.2101), (13.0543, 80.2627),
    (13.0339, 80.2686), (13.0067, 80.2206), (12.9815, 80.2209),
    (12.9010, 80.2279), (13.1167, 80.2333), (12.9516, 80.1462),
    (13.0389, 80.1567)
]

# Sample 1 row per condition label to get variety, then fill to 10
b_samples = []
for label in [0, 1, 2]:
    subset = building_raw[building_raw["Condition Label"] == label]
    n = min(len(subset), 4 if label == 0 else 3)
    b_samples.append(subset.sample(n, random_state=42))
b_samples = pd.concat(b_samples).head(10).reset_index(drop=True)

buildings = []
for i, row in b_samples.iterrows():
    if i >= 10:
        break
    lat, lng = building_coords[i]
    accel = round((abs(float(row["Accel_X (m/s^2)"])) + abs(float(row["Accel_Y (m/s^2)"]))) / 2, 4)
    strain = round(float(row["Strain (με)"]), 2)
    temp = round(float(row["Temp (°C)"]), 2)
    condition = int(row["Condition Label"])
    buildings.append({
        "asset_id": f"BL_{str(i+1).zfill(3)}",
        "name": building_names[i],
        "type": "building",
        "lat": lat, "lng": lng,
        "vibration": accel,
        "crack_index": None, "load_tons": None,
        "temperature": temp,
        "wind_speed": None, "flicker_count": None,
        "hours_offline": None,
        "tilt_angle": None, "co2": None, "humidity": None,
        "strain": strain,
        "pressure": None, "flow_rate": None,
        "leakage_flag": None, "defect_count": None, "avg_confidence": None,
        "_condition_label": condition  # will use in classify
    })

# ── 4. PIPELINES ─────────────────────────────────────────────
pipe_raw = pd.read_csv("location_aware_gis_leakage_dataset.csv")

pipeline_names = [
    "Adyar Water Main", "T Nagar Pipeline", "Anna Nagar Supply Line",
    "Velachery Distribution", "OMR Industrial Pipe",
    "Tambaram Water Main", "Porur Supply Line", "Guindy Pipeline",
    "Chromepet Distribution", "Perambur Water Main"
]
pipeline_coords = [
    (13.0067, 80.2206), (13.0418, 80.2341), (13.0827, 80.2101),
    (12.9815, 80.2209), (12.9010, 80.2279), (12.9229, 80.1275),
    (13.0389, 80.1567), (13.0067, 80.2206), (12.9516, 80.1462),
    (13.1167, 80.2333)
]

# Get variety — some with leakage, some without
leak = pipe_raw[pipe_raw["Leakage_Flag"] == 1].head(4)
no_leak = pipe_raw[pipe_raw["Leakage_Flag"] == 0].head(6)
pipe_samples = pd.concat([leak, no_leak]).reset_index(drop=True)

pipelines = []
for i, row in pipe_samples.iterrows():
    lat, lng = pipeline_coords[i]
    pipelines.append({
        "asset_id": f"PL_{str(i+1).zfill(3)}",
        "name": pipeline_names[i],
        "type": "pipeline",
        "lat": lat, "lng": lng,
        "vibration": round(float(row["Vibration"]), 4),
        "crack_index": None, "load_tons": None,
        "temperature": round(float(row["Temperature"]), 2),
        "wind_speed": None, "flicker_count": None,
        "hours_offline": None, "tilt_angle": None,
        "co2": None, "humidity": None, "strain": None,
        "pressure": round(float(row["Pressure"]), 2),
        "flow_rate": round(float(row["Flow_Rate"]), 2),
        "leakage_flag": int(row["Leakage_Flag"]),
        "defect_count": None, "avg_confidence": None
    })

# ── 5. ROADS ─────────────────────────────────────────────────
road_raw = pd.read_csv("road_health_report.csv")

road_names = [
    "Anna Salai Stretch", "OMR Phase 1", "GST Road Section",
    "ECR Stretch", "Poonamallee Highway", "Mount Road Section",
    "Inner Ring Road", "Rajiv Gandhi Salai", "NH48 Chennai",
    "Velachery Main Road"
]
road_coords = [
    (13.0569, 80.2425), (12.9279, 80.2157), (12.9715, 80.1959),
    (12.8406, 80.2323), (13.0469, 80.1165), (13.0569, 80.2425),
    (13.0418, 80.2101), (12.9010, 80.2279), (13.0826, 80.2101),
    (12.9815, 80.2209)
]

# Sample: Bad, Moderate, and good (low defect_count)
bad = road_raw[road_raw["health_status"] == "Bad"].head(4)
moderate = road_raw[road_raw["health_status"] == "Moderate"].head(3)
good = road_raw[road_raw["defect_count"] == 0].head(3) if len(road_raw[road_raw["defect_count"] == 0]) >= 3 else road_raw.tail(3)
road_samples = pd.concat([bad, moderate, good]).head(10).reset_index(drop=True)

roads = []
for i, row in road_samples.iterrows():
    lat, lng = road_coords[i]
    roads.append({
        "asset_id": f"RD_{str(i+1).zfill(3)}",
        "name": road_names[i],
        "type": "road",
        "lat": lat, "lng": lng,
        "vibration": None, "crack_index": None, "load_tons": None,
        "temperature": None, "wind_speed": None, "flicker_count": None,
        "hours_offline": None, "tilt_angle": None, "co2": None,
        "humidity": None, "strain": None, "pressure": None,
        "flow_rate": None,
        "leakage_flag": None,
        "defect_count": int(row["defect_count"]),
        "avg_confidence": round(float(row["avg_confidence"]), 3),
        "_road_status": str(row["health_status"])
    })

# ── 6. COMBINE & SAVE ────────────────────────────────────────
all_assets = bridges + streetlights + buildings + pipelines + roads
df = pd.DataFrame(all_assets)

# Drop internal helper columns before saving
df = df.drop(columns=[c for c in ["_condition_label", "_road_status"] if c in df.columns])

df.to_csv("master_assets.csv", index=False)
print(f"Done! {len(df)} assets saved to master_assets.csv")
print(df["type"].value_counts())

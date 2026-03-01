import pandas as pd


def classify_bridge(row):
    reasons = []
    vibration = float(row["vibration"]) if pd.notna(row["vibration"]) else 0
    crack = float(row["crack_index"]) if pd.notna(row["crack_index"]) else 0
    if vibration > 8.0 or crack > 7.0:
        if vibration > 8.0: reasons.append(f"Vibration {vibration} m/s² exceeds critical limit of 8.0")
        if crack > 7.0: reasons.append(f"Crack index {crack} exceeds critical limit of 7.0")
        return "Critical", reasons
    elif vibration > 5.0 or crack > 4.0:
        if vibration > 5.0: reasons.append(f"Vibration {vibration} m/s² exceeds warning limit of 5.0")
        if crack > 4.0: reasons.append(f"Crack index {crack} exceeds warning limit of 4.0")
        return "Warning", reasons
    return "Healthy", ["All structural readings within normal range"]


def classify_streetlight(row):
    reasons = []
    flicker = float(row["flicker_count"]) if pd.notna(row["flicker_count"]) else 0
    offline = float(row["hours_offline"]) if pd.notna(row["hours_offline"]) else 0
    if offline > 48 or flicker >= 5:
        if offline > 48: reasons.append(f"Offline for {offline} hrs — exceeds critical limit of 48 hrs")
        if flicker >= 5: reasons.append(f"Flicker count {int(flicker)} — exceeds critical limit of 5")
        return "Critical", reasons
    elif offline > 24 or flicker >= 3:
        if offline > 24: reasons.append(f"Offline for {offline} hrs — exceeds warning limit of 24 hrs")
        if flicker >= 3: reasons.append(f"Flicker count {int(flicker)} — exceeds warning limit of 3")
        return "Warning", reasons
    return "Healthy", ["Light operating normally"]


def classify_building(row):
    reasons = []
    # Use real Condition Label from dataset if available
    label_col = "condition_label" if "condition_label" in row.index else None
    vibration = float(row["vibration"]) if pd.notna(row.get("vibration", None)) else 0
    strain = float(row["strain"]) if pd.notna(row.get("strain", None)) else 0

    # Classify by strain and vibration thresholds
    if strain > 180 or vibration > 0.8:
        if strain > 180: reasons.append(f"Structural strain {strain} με exceeds critical limit of 180")
        if vibration > 0.8: reasons.append(f"Acceleration {vibration} m/s² indicates severe vibration")
        return "Critical", reasons
    elif strain > 120 or vibration > 0.4:
        if strain > 120: reasons.append(f"Structural strain {strain} με exceeds warning limit of 120")
        if vibration > 0.4: reasons.append(f"Acceleration {vibration} m/s² indicates elevated vibration")
        return "Warning", reasons
    return "Healthy", ["Structural readings within safe parameters"]


def classify_pipeline(row):
    reasons = []
    leakage = int(row["leakage_flag"]) if pd.notna(row.get("leakage_flag", None)) else 0
    pressure = float(row["pressure"]) if pd.notna(row.get("pressure", None)) else 0
    vibration = float(row["vibration"]) if pd.notna(row.get("vibration", None)) else 0

    if leakage == 1:
        reasons.append("Active leakage detected by sensor")
        return "Critical", reasons
    elif pressure > 80 or vibration > 4.0:
        if pressure > 80: reasons.append(f"Pressure {pressure} bar exceeds warning limit of 80")
        if vibration > 4.0: reasons.append(f"Pipeline vibration {vibration} exceeds warning limit of 4.0")
        return "Warning", reasons
    return "Healthy", ["No leakage detected, pressure nominal"]


def classify_road(row):
    reasons = []
    defects = int(row["defect_count"]) if pd.notna(row.get("defect_count", None)) else 0
    confidence = float(row["avg_confidence"]) if pd.notna(row.get("avg_confidence", None)) else 0

    if defects >= 5 or (defects >= 3 and confidence > 0.7):
        reasons.append(f"YOLOv8 detected {defects} defects with {round(confidence*100)}% avg confidence")
        return "Critical", reasons
    elif defects >= 2 or (defects >= 1 and confidence > 0.5):
        reasons.append(f"YOLOv8 detected {defects} defect(s) with {round(confidence*100)}% avg confidence")
        return "Warning", reasons
    return "Healthy", ["No significant road defects detected by AI model"]


def classify_asset(row):
    t = row["type"]
    if t == "bridge":      return classify_bridge(row)
    if t == "streetlight": return classify_streetlight(row)
    if t == "building":    return classify_building(row)
    if t == "pipeline":    return classify_pipeline(row)
    if t == "road":        return classify_road(row)
    return "Unknown", ["Unknown asset type"]


def run_classification():
    df = pd.read_csv("master_assets.csv")
    results = df.apply(classify_asset, axis=1)
    df["health_status"] = results.apply(lambda x: x[0])
    df["reason"] = results.apply(lambda x: ", ".join(x[1]))
    df.to_csv("master_assets.csv", index=False)
    print("Classification done!")
    print(df.groupby(["type", "health_status"]).size().unstack(fill_value=0))
    return df


if __name__ == "__main__":
    run_classification()

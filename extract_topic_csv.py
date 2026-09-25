#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "disaster_narratives.csv"

TOPIC_TEMPLATES = {
    "flood": [
        "heavy rain flooded homes and roads",
        "river overflow forced evacuation across the district",
        "flash flooding closed schools and highways",
        "storm drains overflowed during severe rainfall",
        "low-lying neighborhoods were inundated after a flood warning",
        "floodwater damaged homes and disrupted public transport",
        "towering floodwaters reached the market area and shelter centers",
        "residents evacuated as flood conditions worsened overnight",
    ],
    "earthquake": [
        "earthquake tremors shook multiple neighborhoods",
        "quake damage knocked down power lines and buildings",
        "residents ran outside after a strong earthquake warning",
        "aftershocks caused panic in city districts",
        "earthquake cracks appeared in roads and bridges",
        "structural damage forced inspections across the region",
        "an earthquake impact report showed widespread debris",
        "sudden seismic activity damaged homes and utilities",
    ],
    "fire": [
        "forest fire spread rapidly across dry hills",
        "wildfire threatened villages and nearby farms",
        "firefighters fought an intense wildfire overnight",
        "grass fire caused smoke and road closures",
        "blaze spread through a warehouse district",
        "fire damaged homes and damaged emergency access routes",
        "smoke from the wildfire spread across the town",
        "fire crews evacuated residents near the burning area",
    ],
    "storm": [
        "storm winds toppled trees and power cables",
        "severe storm warnings were issued for coastal zones",
        "gusty winds damaged roofs and roadside signs",
        "storm surges threatened local harbors",
        "hailstorm disrupted traffic and agriculture",
        "strong winds caused widespread outages in the city",
        "storm conditions damaged infrastructure and farms",
        "cyclonic winds knocked out public utilities",
    ],
    "drought": [
        "drought conditions drained irrigation channels",
        "long dry spell left farms without water",
        "water scarcity threatened communities during drought",
        "drought damaged rice fields and livestock supply",
        "low rainfall created severe drought warnings",
        "dry conditions reduced crop yields in the valley",
        "drought conditions increased wildfire risk",
        "water reservoirs dropped sharply during the drought",
    ],
    "landslide": [
        "landslide blocked roads after heavy rainfall",
        "mudslide swept through a hillside community",
        "landslides damaged homes along the mountain slope",
        "slope failure buried a bridge and cut access",
        "rain-induced landslide closed a major route",
        "debris flow from landslide reached a nearby village",
        "landslides disrupted transport and emergency crews",
        "hillside collapse left families without shelter",
    ],
}

def build_default_dataset() -> pd.DataFrame:
    records = []
    topics = [
        ("Flood", "flood"),
        ("Earthquake", "earthquake"),
        ("Fire", "fire"),
        ("Storm", "storm"),
        ("Drought", "drought"),
        ("Landslide", "landslide"),
    ]
    for idx in range(50):
        disaster_type, topic_key = topics[idx % len(topics)]
        template = TOPIC_TEMPLATES[topic_key][idx % len(TOPIC_TEMPLATES[topic_key])]
        location = [
            "Metro City",
            "North Valley",
            "Coastal Bay",
            "Ridge Town",
            "Central Plains",
            "Hill District",
            "West Delta",
            "Riverbend",
            "Harbor Point",
            "Forest Edge",
        ][idx % 10]
        severity = ["Low", "Medium", "High", "Critical"][idx % 4]
        title = f"{disaster_type} event {idx + 1}"
        records.append(
            {
                "id": idx + 1,
                "title": title,
                "location": location,
                "disaster_type": disaster_type,
                "severity": severity,
                "narrative": template + f" in {location} during the recent emergency.",
                "source": "Hugging Face dataset export",
            }
        )
    return pd.DataFrame(records)


def ensure_dataset_exists() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        df = build_default_dataset()
        df.to_csv(DATASET_PATH, index=False)
        return df
    return pd.read_csv(DATASET_PATH)


def extract_topic(topic: str) -> pd.DataFrame:
    df = ensure_dataset_exists()
    topic_value = topic.strip().lower()
    if "all" == topic_value:
        return df
    filtered = df[df["disaster_type"].astype(str).str.lower().str.contains(topic_value, na=False)].copy()
    if filtered.empty:
        filtered = df[df["narrative"].astype(str).str.lower().str.contains(topic_value, na=False)].copy()
    return filtered.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a CSV subset from the disaster dataset by topic.")
    parser.add_argument("--topic", default="Flood", help="Topic or disaster type to filter by, for example Flood, Fire, Earthquake.")
    parser.add_argument("--output", default=None, help="Optional output CSV path. Defaults to data/<topic>_narratives.csv.")
    args = parser.parse_args()

    filtered = extract_topic(args.topic)
    output_path = Path(args.output) if args.output else DATA_DIR / f"{args.topic.strip().lower()}_narratives.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    print(f"Rows extracted: {len(filtered)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

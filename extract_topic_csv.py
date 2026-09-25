#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "disaster_narratives.csv"
ALL_DATASET_PATH = DATA_DIR / "all_narratives.csv"

MESSAGES_URL = "https://raw.githubusercontent.com/canaveensetia/udacity-disaster-response-pipeline/master/data/disaster_messages.csv"
CATEGORIES_URL = "https://raw.githubusercontent.com/canaveensetia/udacity-disaster-response-pipeline/master/data/disaster_categories.csv"

DISASTER_TYPES = {
    "floods": "Flood",
    "storm": "Storm",
    "fire": "Fire",
    "earthquake": "Earthquake",
    "cold": "Cold weather",
    "other_weather": "Other weather",
    "weather_related": "Weather-related",
}


def download_online_dataset() -> pd.DataFrame:
    messages = pd.read_csv(MESSAGES_URL)
    categories = pd.read_csv(CATEGORIES_URL)
    category_names = [part.rsplit("-", 1)[0] for part in categories["categories"].iloc[0].split(";")]
    category_values = categories["categories"].str.split(";", expand=True)
    category_values.columns = category_names
    category_values.index = categories["id"]
    for column in category_values.columns:
        category_values[column] = category_values[column].str.rsplit("-", n=1).str[-1].astype(int)

    merged = messages.merge(category_values, left_on="id", right_index=True, how="inner")
    available_types = [name for name in DISASTER_TYPES if name in merged.columns]

    def classify_disaster(row: pd.Series) -> str:
        for category in available_types:
            if row[category] == 1:
                return DISASTER_TYPES[category]
        return "General disaster response"

    severity_columns = [
        column
        for column in [
            "aid_related",
            "medical_help",
            "search_and_rescue",
            "death",
            "infrastructure_related",
            "weather_related",
        ]
        if column in merged.columns
    ]
    severity_score = merged[severity_columns].sum(axis=1) if severity_columns else 0
    merged["severity"] = pd.cut(
        severity_score,
        bins=[-1, 0, 1, 2, float("inf")],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    result = pd.DataFrame(
        {
            "id": merged["id"].astype(int),
            "title": "Disaster message " + merged["id"].astype(str),
            "location": "Not specified",
            "disaster_type": merged.apply(classify_disaster, axis=1),
            "severity": merged["severity"],
            "narrative": merged["message"].fillna("").astype(str).str.strip(),
            "source": "Udacity/Figure Eight Disaster Response dataset",
        }
    )
    return result[result["narrative"] != ""].drop_duplicates("id").reset_index(drop=True)


def ensure_dataset_exists(refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if refresh or not DATASET_PATH.exists():
        df = download_online_dataset()
        df.to_csv(DATASET_PATH, index=False)
        df.to_csv(ALL_DATASET_PATH, index=False)
        return df
    return pd.read_csv(DATASET_PATH)


def extract_topic(topic: str, refresh: bool = False) -> pd.DataFrame:
    df = ensure_dataset_exists(refresh=refresh)
    topic_value = topic.strip().lower()
    if topic_value == "all":
        return df
    filtered = df[df["disaster_type"].astype(str).str.lower().str.contains(topic_value, na=False)].copy()
    if filtered.empty:
        filtered = df[df["narrative"].astype(str).str.lower().str.contains(topic_value, na=False)].copy()
    return filtered.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and filter the online disaster-response dataset.")
    parser.add_argument("--topic", default="Flood", help="Topic or disaster type to filter.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    parser.add_argument("--refresh", action="store_true", help="Download the online dataset again.")
    args = parser.parse_args()

    filtered = extract_topic(args.topic, refresh=args.refresh)
    output_path = Path(args.output) if args.output else DATA_DIR / f"{args.topic.strip().lower()}_narratives.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    print(f"Rows extracted: {len(filtered)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

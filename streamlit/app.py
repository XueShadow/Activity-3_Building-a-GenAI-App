from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

project_dir = Path(__file__).resolve().parents[1]
for dotenv_path in [project_dir / ".env", project_dir.parent / ".env"]:
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

DATA_PATH = project_dir / "data" / "disaster_narratives.csv"
FALLBACK_DATA_PATH = project_dir / "data" / "all_narratives.csv"


def load_data() -> pd.DataFrame:
    csv_path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    if not csv_path.exists():
        return pd.DataFrame(columns=["title", "region", "disaster_type", "severity", "summary"])
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame(columns=["title", "region", "disaster_type", "severity", "summary"])
        return df
    except Exception:
        return pd.DataFrame(columns=["title", "region", "disaster_type", "severity", "summary"])


def normalize_severity(value):
    if pd.isna(value):
        return "Not specified"
    text = str(value).strip().lower()
    try:
        score = float(text)
        if score >= 9:
            return "Critical"
        if score >= 7:
            return "High"
        if score >= 4:
            return "Medium"
        return "Low"
    except ValueError:
        pass
    mapping = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
    return mapping.get(text, text.title())


def parse_date_value(value):
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "not specified"}:
        return pd.NaT
    try:
        return pd.to_datetime(text, errors="raise")
    except Exception:
        pass

    match = re.search(r"(\d{4})[-_/]?(\d{2})[-_/]?(\d{2})(?:[_ -]?(\d{2})(?:\d{2})?)?", text)
    if match:
        year, month, day, hour = match.groups()
        dt_text = f"{year}-{month or '01'}-{day or '01'}"
        if hour:
            dt_text = f"{dt_text} {hour}:00:00"
        try:
            return pd.to_datetime(dt_text)
        except Exception:
            pass
    return pd.NaT


def enrich_narrative(row: pd.Series) -> str:
    details = []
    notes = row.get("notes") or row.get("narrative") or row.get("summary") or row.get("description")
    if notes and str(notes).strip() and str(notes).strip().lower() not in {"nan", "none", "not specified"}:
        details.append(str(notes).strip())
    tags = row.get("tags") or row.get("keywords")
    if tags and str(tags).strip() and str(tags).strip().lower() not in {"nan", "none", "not specified"}:
        details.append(str(tags).strip())
    outbreak = row.get("outbreak")
    if outbreak and str(outbreak).strip() and str(outbreak).strip() not in {"*", "nan", "none", "not specified"}:
        details.append(str(outbreak).strip())
    status = row.get("status indicator") or row.get("status_indicator") or row.get("status")
    if status and str(status).strip() and str(status).strip().lower() not in {"nan", "none", "not specified"}:
        details.append(str(status).strip())
    if not details:
        return "No detailed narrative available."
    return " ".join(details)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    normalized = cleaned.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    legacy_columns = {"title", "region", "disaster_type", "severity", "summary"}
    if legacy_columns.issubset(normalized.columns) and not {"narrative", "location"}.issubset(normalized.columns):
        normalized = normalized.rename(columns={"region": "location", "summary": "narrative"})
        normalized["id"] = range(1, len(normalized) + 1)
        normalized["source"] = normalized["title"]

    if "id" not in normalized.columns:
        normalized["id"] = range(1, len(normalized) + 1)

    date_candidates = [column for column in normalized.columns if "date" in column.lower() or "time" in column.lower()]
    if date_candidates:
        normalized["date"] = normalized[date_candidates[0]].apply(parse_date_value)
    elif "title" in normalized.columns:
        normalized["date"] = normalized["title"].apply(parse_date_value)
    else:
        normalized["date"] = pd.NaT

    if "location" not in normalized.columns:
        normalized["location"] = "Not specified"
    if "disaster_type" not in normalized.columns:
        normalized["disaster_type"] = "Not specified"
    if "narrative" not in normalized.columns:
        normalized["narrative"] = normalized.get("summary", "")
    if "source" not in normalized.columns:
        normalized["source"] = "Dataset"
    if "severity" not in normalized.columns:
        normalized["severity"] = "Medium"

    normalized["narrative"] = normalized.apply(enrich_narrative, axis=1)

    required = ["id", "date", "location", "disaster_type", "narrative", "source", "severity"]
    for column in required:
        if column not in normalized.columns:
            return pd.DataFrame(columns=required)

    normalized = normalized.dropna(axis=0, how="all")
    normalized = normalized[normalized["narrative"].fillna("").astype(str).str.strip() != ""]
    normalized["location"] = normalized["location"].fillna("Not specified").astype(str).str.strip().replace({"": "Not specified"})
    normalized["disaster_type"] = normalized["disaster_type"].fillna("Not specified").astype(str).str.strip().replace({"": "Not specified"})
    normalized["source"] = normalized["source"].fillna("Not specified").astype(str).str.strip().replace({"": "Not specified"})
    normalized["severity"] = normalized["severity"].apply(normalize_severity)
    return normalized.reset_index(drop=True)


def persist_dataset(dataset: pd.DataFrame) -> None:
    if dataset is None or dataset.empty:
        return

    saved = clean_data(dataset.copy())
    if saved.empty:
        return

    columns = ["id", "date", "location", "disaster_type", "narrative", "source", "severity"]
    for column in columns:
        if column not in saved.columns:
            saved[column] = ""
    saved = saved[columns].copy()
    saved["date"] = pd.to_datetime(saved["date"], errors="coerce")

    save_paths = [project_dir / "data" / "disaster_narratives.csv", project_dir / "data" / "all_narratives.csv"]
    for path in save_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        saved.to_csv(path, index=False)


def save_analyzed_narrative(narrative: str, result: dict) -> None:
    text = (narrative or "").strip()
    if not text:
        return

    session_state = st.session_state
    if isinstance(session_state, dict):
        existing = session_state.get("dataset", pd.DataFrame())
    else:
        existing = getattr(session_state, "dataset", pd.DataFrame())

    if isinstance(existing, pd.DataFrame) and not existing.empty:
        duplicate_mask = existing["narrative"].astype(str).str.strip().str.lower() == text.lower()
        if duplicate_mask.any():
            return

    new_row = {
        "id": int(existing["id"].max()) + 1 if isinstance(existing, pd.DataFrame) and not existing.empty and "id" in existing.columns else 1,
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "location": result.get("location") or "Not specified",
        "disaster_type": result.get("disaster_type") or "Not specified",
        "narrative": text,
        "source": "Narrative Analyzer",
        "severity": normalize_severity(result.get("severity") or "Not specified"),
    }

    new_df = pd.DataFrame([new_row])
    merged = pd.concat([existing, new_df], ignore_index=True) if isinstance(existing, pd.DataFrame) and not existing.empty else new_df
    dataset = clean_data(merged)

    if isinstance(session_state, dict):
        session_state["dataset"] = dataset
    else:
        session_state.dataset = dataset

    persist_dataset(dataset)


def get_api_key() -> str:
    return os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "") or os.getenv("OPENAI_API_KEY", "")


def heuristic_analysis(narrative: str) -> dict:
    lower = narrative.lower()
    disaster_types = ["Flood", "Typhoon", "Landslide", "Earthquake", "Storm Surge", "Fire", "Drought", "Flash Flood"]
    for disaster in disaster_types:
        if disaster.lower() in lower:
            return {
                "disaster_type": disaster,
                "location": "Not specified",
                "severity": "High" if any(word in lower for word in ["severe", "heavy", "critical"]) else "Medium",
                "urgency": "Urgent" if any(word in lower for word in ["evacuate", "urgent", "immediately"]) else "Moderate",
                "summary": narrative[:200],
                "impacts": ["Flooding", "Road disruption"],
                "response_actions": ["Evacuation", "Emergency response"],
                "keywords": [word for word in lower.split() if len(word) > 4][:6],
            }
    return {
        "disaster_type": "Not specified",
        "location": "Not specified",
        "severity": "Not specified",
        "urgency": "Not specified",
        "summary": narrative[:200],
        "impacts": [],
        "response_actions": [],
        "keywords": [word for word in lower.split() if len(word) > 4][:6],
    }


def analyze_narrative(narrative: str) -> dict:
    api_key = get_api_key()
    if not api_key:
        return heuristic_analysis(narrative)

    try:
        client = InferenceClient(token=api_key)
        prompt = "You are a disaster narrative assistant. Extract facts only. Return valid JSON with keys disaster_type, location, severity, urgency, summary, impacts, response_actions, keywords. Use 'Not specified' when information is unavailable."
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": narrative},
            ],
            model=os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            max_tokens=400,
            temperature=0.2,
        )
        text = str(response.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`").replace("json", "", 1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    return heuristic_analysis(narrative)


def format_analysis_field(value: object) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items) if items else "Not specified"
    text = str(value).replace("\n", " ").strip()
    return text or "Not specified"


def answer_dataset_question(question: str, dataset: pd.DataFrame) -> tuple[str, str]:
    api_key = get_api_key()
    if api_key:
        try:
            client = InferenceClient(token=api_key)
            context = dataset.head(15).to_dict(orient="records")
            prompt = (
                "Use the dataset records below to answer the user's question. "
                "Answer only with the final response text, no markdown and no code fences. "
                f"Dataset: {json.dumps(context, ensure_ascii=False)}\n"
                f"Question: {question}"
            )
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a dataset QA assistant. Use only the provided dataset to answer the question."},
                    {"role": "user", "content": prompt},
                ],
                model=os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                max_tokens=250,
                temperature=0.2,
            )
            text = str(response.choices[0].message.content or "").strip()
            if text:
                if text.startswith("```"):
                    text = text.strip("`").replace("json", "", 1).strip()
                return text, "Hugging Face model"
        except Exception:
            pass

    question_lower = question.lower()
    if "how many" in question_lower or "total" in question_lower:
        return f"The dataset contains {len(dataset)} narrative(s).", "Dataset fallback"
    if "location" in question_lower or "most reports" in question_lower:
        counts = dataset["location"].value_counts()
        location = counts.index[0] if not counts.empty else "Not specified"
        count = int(counts.iloc[0]) if not counts.empty else 0
        return f"The location with the most reports is {location} with {count} narrative(s).", "Dataset fallback"
    if "severity" in question_lower:
        return f"Severity distribution: {dataset['severity'].value_counts().to_dict()}", "Dataset fallback"
    if "flood" in question_lower:
        count = int(dataset["disaster_type"].astype(str).str.lower().str.contains("flood").sum())
        return f"The dataset contains {count} flood-related narrative(s).", "Dataset fallback"
    return "Ask about total narratives, disaster types, locations, or severity distribution.", "Dataset fallback"


st.set_page_config(page_title="AI DisasterLens", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(180deg, #f5f7ff 0%, #eef4ff 100%);
        }

        .stApp {
            background: linear-gradient(180deg, #f8faff 0%, #eef4ff 100%);
        }

        div[data-testid="stAppViewContainer"] {
            background: transparent;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        .hero-box {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 48%, #2563eb 100%);
            border-radius: 20px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.22);
        }

        .hero-box h1 {
            color: white !important;
            margin: 0 0 0.2rem 0;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.05em;
        }

        .hero-box p {
            color: rgba(255,255,255,0.85);
            margin: 0;
            font-size: 1rem;
        }

        .stMetric {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }

        .stMetric [data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 600;
        }

        .stMetric [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.1rem;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > div,
        .stDataFrame {
            border-radius: 12px !important;
            border: 1px solid rgba(148, 163, 184, 0.5) !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }

        .stDataFrame {
            background: rgba(255,255,255,0.9);
        }

        .sidebar-content {
            background: rgba(248, 250, 252, 0.9);
        }

        section[data-testid="stSidebar"] > div {
            background: rgba(248, 250, 252, 0.9);
        }

        h2 {
            color: #0f172a;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        .stSuccess, .stInfo, .stWarning {
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
        <h1>AI DisasterLens</h1>
        <p>A modern disaster narrative dashboard for exploring incidents, monitoring patterns, and extracting insight from event reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "dataset" not in st.session_state:
    st.session_state.dataset = clean_data(load_data())

df = st.session_state.dataset.copy()

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload dataset", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            cleaned = clean_data(uploaded_df)
            if not cleaned.empty:
                existing = st.session_state.get("dataset", pd.DataFrame())
                if existing.empty:
                    st.session_state.dataset = cleaned
                else:
                    merged = pd.concat([existing, cleaned], ignore_index=True)
                    st.session_state.dataset = clean_data(merged)
                df = st.session_state.dataset.copy()
                st.success("Dataset uploaded and appended to the existing data.")
            else:
                st.warning("The uploaded CSV did not contain usable data.")
        except Exception as exc:
            st.error(f"Upload error: {exc}")

    st.subheader("Filters")
    if not df.empty:
        chosen_type = st.selectbox("Disaster type", ["All"] + sorted(df["disaster_type"].dropna().unique().tolist()))
        chosen_location = st.selectbox("Location", ["All"] + sorted(df["location"].dropna().unique().tolist()))
    else:
        chosen_type = "All"
        chosen_location = "All"

if df.empty:
    st.warning("No dataset is available. Upload a CSV file or check the default data file.")
else:
    if chosen_type != "All":
        df = df[df["disaster_type"] == chosen_type]
    if chosen_location != "All":
        df = df[df["location"] == chosen_location]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total narratives", len(df))
    col2.metric("Disaster types", df["disaster_type"].nunique())
    col3.metric("Locations", df["location"].nunique())
    col4.metric("High/Critical", int((df["severity"].isin(["High", "Critical"])).sum()))

    charts_col1, charts_col2 = st.columns(2)
    with charts_col1:
        disaster_counts = df["disaster_type"].value_counts().reset_index()
        disaster_counts.columns = ["disaster_type", "count"]
        st.plotly_chart(px.bar(disaster_counts, x="disaster_type", y="count", title="Disaster type distribution"), width="stretch")

    with charts_col2:
        severity_counts = df["severity"].value_counts().reset_index()
        severity_counts.columns = ["severity", "count"]
        st.plotly_chart(px.pie(severity_counts, names="severity", values="count", title="Severity distribution"), width="stretch")

    st.subheader("Recent narratives")
    recent_df = df.sort_values(by=["date", "id"], ascending=[False, False], kind="mergesort").head(10)
    st.dataframe(recent_df[["id", "date", "location", "disaster_type", "severity", "narrative"]], width="stretch")

    st.subheader("Narrative analyzer")
    narrative = st.text_area("Paste a disaster narrative here...")
    if st.button("Analyze narrative") and narrative.strip():
        result = analyze_narrative(narrative)
        save_analyzed_narrative(narrative, result)
        st.success("Analyzed narrative saved to the dataset and CSV files.")
        st.subheader("Analysis result")
        st.write(f"Disaster Type: **{result.get('disaster_type', 'Not specified')}**")
        st.write(f"Location: **{result.get('location', 'Not specified')}**")
        st.write(f"Severity: **{result.get('severity', 'Not specified')}**")
        st.write(f"Urgency: **{result.get('urgency', 'Not specified')}**")
        st.write(f"Summary: **{format_analysis_field(result.get('summary'))}**")
        st.write(f"Impacts: **{format_analysis_field(result.get('impacts'))}**")
        st.write(f"Response Actions: **{format_analysis_field(result.get('response_actions'))}**")
        st.write(f"Keywords: **{format_analysis_field(result.get('keywords'))}**")



from __future__ import annotations

import json
import os
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = df.copy()
    cleaned.columns = [str(column).strip().lower() for column in cleaned.columns]

    legacy_columns = {"title", "region", "disaster_type", "severity", "summary"}
    if legacy_columns.issubset(cleaned.columns) and not {"narrative", "location"}.issubset(cleaned.columns):
        cleaned = cleaned.rename(columns={"region": "location", "summary": "narrative"})
        cleaned["id"] = range(1, len(cleaned) + 1)
        cleaned["date"] = pd.NaT
        cleaned["source"] = cleaned["title"]

    if "id" not in cleaned.columns:
        cleaned["id"] = range(1, len(cleaned) + 1)
    if "date" not in cleaned.columns:
        cleaned["date"] = pd.NaT
    if "location" not in cleaned.columns:
        cleaned["location"] = "Not specified"
    if "disaster_type" not in cleaned.columns:
        cleaned["disaster_type"] = "Not specified"
    if "narrative" not in cleaned.columns:
        cleaned["narrative"] = cleaned.get("summary", "")
    if "source" not in cleaned.columns:
        cleaned["source"] = "Dataset"
    if "severity" not in cleaned.columns:
        cleaned["severity"] = "Medium"

    required = ["id", "date", "location", "disaster_type", "narrative", "source", "severity"]
    for column in required:
        if column not in cleaned.columns:
            return pd.DataFrame(columns=required)

    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = cleaned[cleaned["narrative"].fillna("").astype(str).str.strip() != ""]
    cleaned["location"] = cleaned["location"].fillna("Not specified").astype(str).str.strip().replace({"": "Not specified"})
    cleaned["disaster_type"] = cleaned["disaster_type"].fillna("Not specified").astype(str).str.strip().replace({"": "Not specified"})
    cleaned["source"] = cleaned["source"].fillna("Not specified").astype(str).str.strip().replace({"": "Not specified"})
    cleaned["severity"] = cleaned["severity"].apply(normalize_severity)
    return cleaned.reset_index(drop=True)


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
st.title("AI DisasterLens")
st.caption("A disaster narrative analysis app using a different dataset and GenAI-based insights.")
st.markdown("This app reviews disaster narratives, summarizes risk patterns, and answers quick questions from the dataset.")

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
                st.session_state.dataset = cleaned
                df = cleaned.copy()
                st.success("Dataset uploaded and cleaned.")
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
    st.dataframe(df.head(10)[["id", "date", "location", "disaster_type", "severity", "narrative"]], width="stretch")

    st.subheader("Narrative analyzer")
    narrative = st.text_area("Paste a disaster narrative here...")
    if st.button("Analyze narrative") and narrative.strip():
        result = analyze_narrative(narrative)
        st.subheader("Analysis result")
        st.write(f"Disaster Type: **{result.get('disaster_type', 'Not specified')}**")
        st.write(f"Location: **{result.get('location', 'Not specified')}**")
        st.write(f"Severity: **{result.get('severity', 'Not specified')}**")
        st.write(f"Urgency: **{result.get('urgency', 'Not specified')}**")
        st.write(f"Summary: **{format_analysis_field(result.get('summary'))}**")
        st.write(f"Impacts: **{format_analysis_field(result.get('impacts'))}**")
        st.write(f"Response Actions: **{format_analysis_field(result.get('response_actions'))}**")
        st.write(f"Keywords: **{format_analysis_field(result.get('keywords'))}**")

    st.subheader("Dataset chatbot")
    question = st.text_input("Ask about the loaded dataset", placeholder="How many flood narratives are there?")
    if st.button("Ask chatbot") and question.strip():
        answer, _ = answer_dataset_question(question, df)
        st.info(answer)

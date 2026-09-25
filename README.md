# Activity 3 GenAI App

This project follows the CS 315 Activity 3 requirement to build a GenAI app using a different dataset, while keeping the original disaster-analysis idea.

## App idea

A disaster narrative analysis dashboard that reads incident summaries, filters by disaster type/location, and uses GenAI to extract structured insights from local narratives.

## Structure

- data/disaster_narratives.csv
- streamlit/app.py

## Run the app

```bash
cd "Application Development and Emerging Technologies/Activity_3_GenAI_App"
python3 -m venv .venv
source .venv/bin/activate
pip install -r streamlit/requirements.txt
streamlit run streamlit/app.py
```

## Features

- Load and clean the disaster dataset
- Filter by disaster type and location
- Summary metrics and visual charts
- Narrative analysis with Hugging Face AI or a safe local fallback
- Dataset chatbot for quick questions about the loaded data

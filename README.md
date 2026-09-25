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

## Dataset source and citation

The app uses 26,178 non-empty messages from the Udacity Disaster Response dataset,
created in collaboration with Figure Eight. `extract_topic_csv.py` downloads the
messages and category labels, then transforms them into the dashboard schema.

Source files:

- https://github.com/canaveensetia/udacity-disaster-response-pipeline/tree/master/data
- https://github.com/udacity
- https://www.figure-eight.com/

Suggested citation for the submission document:

> Udacity and Figure Eight. (2017). *Disaster Response Messages dataset* [Data set].
> Retrieved from https://github.com/canaveensetia/udacity-disaster-response-pipeline/tree/master/data

The optional narrative analysis uses Hugging Face Inference API models. Cite the
specific model shown in the app's deployment configuration if AI-generated results
are enabled.

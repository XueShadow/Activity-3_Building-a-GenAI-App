Activity 3 – Building a GenAI App

AI Disaster Narrative Analyzer

A Generative AI-powered Streamlit application that analyzes disaster-related messages and narratives. The app transforms disaster response data into useful visual insights and uses AI to extract structured information from individual narratives.

This project was developed for CS 315 – Application Development and Emerging Technologies, Activity 3.

⸻

Project Overview

The goal of this activity is to build a GenAI application using a dataset different from the previous activity.

For this project, the application uses the Udacity Disaster Response Messages dataset, containing disaster-related messages and their associated categories.

The application allows users to:

* Explore disaster response messages
* Filter data by disaster type and location
* View statistical summaries
* Visualize disaster-related categories
* Analyze individual disaster narratives using Generative AI
* Ask questions about the loaded dataset through a chatbot

⸻

App Idea

AI Disaster Narrative Analyzer

The AI Disaster Narrative Analyzer is an interactive dashboard designed to help users understand disaster-related messages.

Instead of manually reading thousands of disaster messages, the application uses data processing, visualization, and Generative AI to identify useful information from the dataset.

Main Purpose

The application focuses on:

* Disaster message exploration
* Category analysis
* Location-based filtering
* Narrative understanding
* AI-generated summaries and insights
* Dataset question answering

⸻

Dataset

The application uses the Udacity Disaster Response Messages dataset, created in collaboration with Figure Eight.

The dataset contains 26,178 non-empty disaster-related messages with associated disaster-response categories.

The dataset is transformed into a simplified CSV format used by the Streamlit dashboard.

Dataset Fields

The application uses information such as:

* Disaster message
* Disaster type
* Location
* Disaster-related categories

Dataset Source

Udacity and Figure Eight. (2017). Disaster Response Messages dataset.

Original dataset repository:

https://github.com/canaveensetia/udacity-disaster-response-pipeline/tree/master/data

⸻

Technologies Used

Programming Language

* Python

Data Processing

* Pandas

Generative AI

* Hugging Face Inference API
* Hugging Face-compatible language models
* Local fallback analysis when AI services are unavailable

Web Application

* Streamlit

Data Visualization

* Streamlit charts
* Pandas

Development Tools

* Git
* GitHub
* Python Virtual Environment

⸻

Project Structure

Activity-3_Building-a-GenAI-App/
│
├── data/
│   └── disaster_narratives.csv
│
├── streamlit/
│   ├── app.py
│   └── requirements.txt
│
├── extract_topic_csv.py
│
├── .gitignore
│
└── README.md

⸻

Application Features

1. Dataset Loading and Cleaning

The application uses Pandas to load the disaster dataset and prepare it for analysis.

The preprocessing stage handles:

* Missing values
* Empty messages
* Category information
* Location information
* Data formatting

⸻

2. Interactive Filtering

Users can filter the dataset based on available disaster information.

Examples include:

* Disaster type
* Location
* Categories

Filtering allows users to focus on specific parts of the dataset instead of analyzing all records at once.

⸻

3. Dataset Statistics

The dashboard provides summary information about the loaded dataset.

Examples include:

* Total number of messages
* Number of disaster categories
* Number of locations
* Distribution of disaster-related messages

⸻

4. Data Visualization

The application converts the dataset into visual representations to make patterns easier to understand.

Visualizations can be used to examine:

* Disaster category distribution
* Message frequency
* Location distribution
* Filtered dataset statistics

⸻

5. GenAI Narrative Analysis

Users can select a disaster-related narrative and send it to the Generative AI analysis component.

The AI can extract structured information and provide an interpretation of the selected narrative.

This demonstrates how GenAI can be combined with traditional data analysis.

⸻

6. Dataset Chatbot

The application includes a chatbot that allows users to ask questions about the loaded dataset.

Example questions:

How many disaster messages are in the dataset?
What disaster categories appear most frequently?
Which locations have the most messages?
What types of disasters are represented?
What information can be found in this dataset?

The chatbot provides a more natural way to interact with the dataset.

⸻

How the Application Works

The application follows this workflow:

                ┌─────────────────────┐
                │   Disaster Dataset  │
                │       CSV File      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Pandas Processing │
                │   & Data Cleaning   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Streamlit UI     │
                │ Filters & Dashboard │
                └───────┬─────┬───────┘
                        │     │
             ┌──────────┘     └──────────┐
             ▼                           ▼
    ┌─────────────────┐        ┌─────────────────┐
    │ Data            │        │ GenAI Analysis  │
    │ Visualization   │        │ & Chatbot       │
    └─────────────────┘        └────────┬────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ AI-Generated    │
                              │ Insights        │
                              └─────────────────┘

⸻

Installation

1. Clone the Repository

git clone https://github.com/XueShadow/Activity-3_Building-a-GenAI-App.git

2. Enter the Project Directory

cd Activity-3_Building-a-GenAI-App

3. Create a Virtual Environment

python3 -m venv .venv

4. Activate the Virtual Environment

Linux / macOS

source .venv/bin/activate

Windows

.venv\Scripts\activate

5. Install Dependencies

pip install -r streamlit/requirements.txt

⸻

Running the Application

Start the Streamlit application with:

streamlit run streamlit/app.py

Streamlit will provide a local URL similar to:

http://localhost:8501

Open the URL in a web browser to use the application.

⸻

GenAI Configuration

The application can use the Hugging Face Inference API for AI-powered narrative analysis.

If GenAI functionality is enabled through an API token, configure the required environment variable before running the application.

Example:

export HF_TOKEN="your_huggingface_token"

For Windows PowerShell:

$env:HF_TOKEN="your_huggingface_token"

Do not commit API keys or access tokens to GitHub.

Use environment variables or Streamlit secrets instead.

⸻

Creating the Dataset

The repository includes:

extract_topic_csv.py

This script is responsible for obtaining the disaster-response data and transforming the messages into the CSV format used by the application.

The generated dataset is placed inside:

data/disaster_narratives.csv

⸻

Testing

The application should be tested using different:

* Disaster types
* Locations
* Narratives
* Dataset questions
* AI prompts

Testing should verify that:

1. The dataset loads correctly.
2. Filters return the expected records.
3. Charts update when filters change.
4. AI analysis handles valid narratives.
5. The application continues functioning when the AI service is unavailable.
6. The chatbot provides relevant answers about the dataset.
7. Missing or empty input does not crash the application.

⸻

Future Improvements

The project can be expanded with additional features.

Advanced Filters

Add filters for:

* Disaster categories
* Locations
* Message length
* Disaster type
* Multiple categories

AI Chatbot Improvements

The chatbot could be enhanced to support:

* More complex dataset questions
* Natural-language data queries
* Automatic statistical analysis
* Follow-up questions
* Context-aware conversations

Advanced GenAI Analysis

Future versions could include:

* Automatic narrative summarization
* Keyword extraction
* Emergency information extraction
* Entity extraction
* Urgency classification
* Resource-request detection
* Geographic information extraction
* Multi-message summarization

Visualization Improvements

Additional visualizations could include:

* Interactive Plotly charts
* Geographic maps
* Category correlation analysis
* Time-based analysis
* Disaster category comparisons

⸻

Deployment

The application can be deployed using Streamlit Community Cloud.

General deployment process:

1. Push the project to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Select:

streamlit/app.py

as the application entry point.
4. Configure required secrets such as the Hugging Face token.
5. Deploy the application.

⸻

Learning Objectives

This project demonstrates how several technologies can be combined to create a GenAI-powered data application.

The main concepts demonstrated are:

* Dataset selection
* Data preprocessing
* Pandas data analysis
* Generative AI integration
* API-based AI inference
* Streamlit application development
* Interactive data visualization
* Natural-language dataset interaction
* AI-assisted narrative analysis
* Application deployment

⸻

Academic Activity Requirements

Requirement	Implementation
Choose a new dataset	Udacity Disaster Response Messages dataset
Load and clean dataset	Pandas
Integrate GenAI	Hugging Face Inference API
Build UI	Streamlit
Interactive components	Filters, selectors, chatbot
Visualize results	Streamlit/Pandas charts
Test and iterate	Dataset, filters, AI prompts
Deployment	Streamlit Community Cloud
Future goals	Advanced filters and AI chatbot

⸻

Project Status

Status: Completed for CS 315 Activity 3

The current version includes dataset processing, interactive filtering, visualization, GenAI-assisted narrative analysis, and dataset chatbot functionality.

⸻

Author

Ji Monsales

BS Computer Science
NEMSU

GitHub: https://github.com/XueShadow

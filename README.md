# CS 315 – Application Development and Emerging Technologies

## Activity 3: Building a GenAI App Using a Different Dataset

# AI DisasterLens

This project was developed for CS 315 – Application Development and Emerging Technologies, Activity 3. It follows the requirement to create another Generative AI application using a different dataset and a different problem domain from the previous activity.

The app is a Streamlit-based dashboard for exploring disaster narratives, filtering by location and disaster type, visualizing trends, and using GenAI to analyze narrative content.

---

## 1. App Idea

The app is designed to help users understand and explore disaster-related narratives from a structured dataset.

Instead of manually reading a large volume of reports, the system allows users to:

- view a cleaned disaster dataset in an interactive dashboard
- filter by location and disaster type
- inspect recent narratives in a table
- analyze narrative text with AI-powered extraction
- ask dataset-related questions using a chatbot
- monitor patterns through summary metrics and charts

This turns raw narrative data into a more accessible and insightful decision-support dashboard.

---

## 2. Dataset Used

The project uses the 2nd Data Release of Project Severe Weather Archive of the Philippines (SWAP DR2), provided on Zenodo.

Dataset source:
https://zenodo.org/records/13730302?utm_source=chatgpt.com

This dataset contains severe weather and disaster narratives from the Philippines, including reports related to hail, tornado, and waterspout events. The raw files were cleaned and transformed into the app’s working CSV files stored in the data folder.

### Included files

- data/all_narratives.csv
- data/disaster_narratives.csv
- raw SWAP DR2 source files in the data folder

---

## 3. What the App Does

The application includes the following features:

### Dataset loading and cleaning
- Loads CSV data into a Pandas DataFrame
- Removes blank or unusable rows
- Normalizes fields like location, severity, disaster type, and narrative
- Keeps the dataset in a structured format for analysis

### Interactive dashboard
- Total narratives summary
- Number of disaster types
- Number of locations
- High/Critical severity count
- Visual charts for disaster distribution and severity breakdown

### Narrative analyzer
- Users can paste a disaster narrative
- The app runs AI analysis to infer or identify:
  - disaster type
  - location
  - severity
  - urgency
  - summary
  - impacts
  - response actions
  - keywords

### Dataset chatbot
- Users can ask natural-language questions like:
  - How many narratives are in the dataset?
  - Which location has the most reports?
  - What is the distribution of disaster types?
  - Which narratives are severe?

### Recent narratives table
- Displays the newest entries first for better UX
- Keeps the dataset and table updated when new narratives are added

---

## 4. Activity Requirements Mapping

This project follows the required Activity 3 instructions:

1. Define Your App Idea  
   The app analyzes severe weather narratives and provides AI-based insights.

2. Set Up Your Environment  
   The project uses a Python virtual environment and a Streamlit app structure.

3. Load and Clean the Dataset  
   Pandas is used to clean and normalize the data before app use.

4. Integrate GenAI for Analysis  
   The app integrates a Hugging Face-compatible GenAI model and includes a fallback analysis flow if the API is unavailable.

5. Build the Streamlit Interface  
   The UI is constructed using Streamlit widgets, filters, metric cards, charts, and text inputs.

6. Visualize the Results  
   Charts and summary cards are generated using Streamlit and Plotly-based visualizations.

7. Test and Iterate  
   The app was tested for dataset loading, blank-row cleanup, filtering, narrative analyzer behavior, and upload/save behavior.

8. Deploy Your App  
   The project is ready to be deployed through Streamlit Community Cloud or another hosting platform.

9. Next Goals  
   Future improvements include additional filters, a stronger dashboard theme, and more advanced AI dataset interactions.

---

## 5. Project Structure

```text
Activity-3_Building-a-GenAI-App/
├── data/
│   ├── all_narratives.csv
│   ├── disaster_narratives.csv
│   ├── SWAP DR2 - Hail Reports.csv
│   ├── SWAP DR2 - Tornado Reports.csv
│   ├── SWAP DR2 - Waterspout Reports.csv
│   ├── Project SWAP Preprint (V1).pdf
│   └── raw archive/source files
├── streamlit/
│   ├── app.py
│   └── requirements.txt
├── .gitignore
├── .venv/
├── README.md
└── sample_append_test.csv
```

---

## 6. Technologies Used

- Python
- Pandas
- Streamlit
- Plotly
- Hugging Face Inference API / model integration
- Git and GitHub

---

## 7. Setup Instructions

### Clone the repository

```bash
git clone https://github.com/XueShadow/Activity-3_Building-a-GenAI-App.git
cd Activity-3_Building-a-GenAI-App
```

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r streamlit/requirements.txt
```

### Run the app

```bash
streamlit run streamlit/app.py
```

Then open the local URL provided by Streamlit, usually:

```text
http://localhost:8501
```

---

## 8. GenAI Configuration

The app can use a Hugging Face token for GenAI analysis.

Example:

```bash
export HF_TOKEN="your_huggingface_token"
```

If the token is not configured, the app falls back to local heuristic analysis so it still works with limited AI support.

---

## 9. Testing and Iteration

The app was tested for:

- successful dataset loading
- blank row removal
- narrative cleaning and formatting
- filtering by disaster type and location
- chart updates
- narrative analysis output
- dataset question answering
- auto-save of analyzed narratives
- newest-first recent table ordering

These checks ensure the app is functional and user-friendly.

---

## 10. Future Improvements

The next goals for the project include:

- additional filters for categories, dates, and severity levels
- enhanced chart interactions
- richer chatbot functionality for deeper dataset questions
- better AI structured extraction for disaster event details
- more advanced dashboard design and UX polish
- deployment to Streamlit Community Cloud

---

## 11. Project Status

Status: Completed for CS 315 Activity 3

This app demonstrates how a GenAI-powered dashboard can be built using a different dataset, cleaned and transformed for analysis, then presented in an interactive Streamlit interface.

---

## 12. References

- 2nd Data Release of Project Severe Weather Archive of the Philippines (SWAP DR2) | Zenodo  
  https://zenodo.org/records/13730302?utm_source=chatgpt.com

- CS 315 – Application Development and Emerging Technologies

---

## 13. Developer

Ji Monsales  
BS Computer Science  
NEMSU

GitHub: https://github.com/XueShadow


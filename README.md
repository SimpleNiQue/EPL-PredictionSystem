# EPL Match Prediction System

This project provides a web application and an API for predicting English Premier League (EPL) match outcomes. It leverages historical match data, machine learning models, and a user-friendly interface to offer predictions and probabilities for upcoming matches.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Prediction Algorithm Details](#prediction-algorithm-details)
  - [Data Collection and Preprocessing](#data-collection-and-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Training and Selection](#model-training-and-selection)
  - [Prediction Analysis](#prediction-analysis)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Features

*   **EPL Match Prediction:** Predicts the outcome of a match (Home Win, Draw, Away Win) based on historical team performance.
*   **Probability Output:** Provides probabilities for each possible outcome (Home Win, Draw, Away Win).
*   **User-Friendly Web Interface:** A clean and interactive web interface for selecting teams and match dates, with dynamic feedback.
*   **Date Picker:** Allows users to specify the date of the match for more context-aware predictions.
*   **Dynamic Loading Spinner with Puns:** A visually appealing spinner with rotating football puns to enhance user experience during prediction analysis.
*   **RESTful API:** A programmatic interface to integrate the prediction service into other applications.
*   **Historical Data Integration:** Utilizes multiple seasons of EPL match data for robust model training.

## Project Structure

```
EPL Prediction system/
├── .gitignore
├── .python-version
├── requirements.txt
├── run.py
├── app/
│   ├── __init__.py
│   ├── config.py             # Configuration settings (e.g., model paths)
│   ├── main.py               # Main FastAPI application entry point
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Pydantic schemas for API request/response validation
│   ├── routes/
│   │   ├── __init__.py
│   │   └── endpoints.py      # API routes and logic for predictions
│   ├── services/
│   │   ├── __init__.py
│   │   └── predictor.py      # Core prediction logic, model loading, and feature calculation
│   ├── static/
│   │   └── styles.css        # CSS for the web interface
│   └── templates/
│       └── index.html        # HTML template for the web interface
├── scripts/
│   ├── EPL WINNER MODEL.ipynb # Jupyter Notebook for model training, evaluation, and saving
│   ├── main.py               # Standalone FastAPI script (for API-only testing)
│   └── data/                 # Directory containing historical match CSV files
│       ├── 2015-16.csv
│       └── ... (other season data)
│   └── models/               # Directory to store trained model and metadata
│       ├── model_metadata.pkl
│       └── premier_league_match_predictor.pkl
└── README.md               # Project documentation
```

## Installation

Follow these steps to set up the project locally:

1.  **Clone the repository:**
    ```bash
    git clone <the-github-repo-link>
    cd EPL-Prediction-system
    ```

2.  **Python Version:**
    This project is developed with Python `3.12.7`. It's recommended to use a virtual environment. You can check your Python version with `python --version` or `python3 --version`. If you use `pyenv`, you can set the local version:
    ```bash
    pyenv install 3.12.7
    pyenv local 3.12.7
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download Historical Data:**
    Ensure you have the historical EPL match data (`.csv` files for each season) in the `scripts/data/` directory. These files are crucial for training the model and for the `predictor.py` service. The notebook `EPL WINNER MODEL.ipynb` expects these files to be present.

6.  **Train the Machine Learning Model:**
    The prediction model needs to be trained and saved.
    *   Open the `scripts/EPL WINNER MODEL.ipynb` Jupyter Notebook.
    *   Run all cells in the notebook. This will:
        *   Load and preprocess historical data.
        *   Perform feature engineering.
        *   Train a `RandomForestClassifier` model (chosen for its performance).
        *   Save the trained model (`premier_league_match_predictor.pkl`) and its metadata (`model_metadata.pkl`) into the `scripts/models/` directory. These files are then used by the `app/services/predictor.py`.

## Usage

### Running the Web Application

To start the web application with the integrated UI:

```bash
python3 run.py
```

The application will typically run on `http://127.0.0.1:8000`. Open this URL in your web browser to access the EPL Match Predictor interface.

### Running the API (Standalone)

If you only want to run the FastAPI backend without the Jinja2 templating for the UI, you can use the `scripts/main.py` file:

```bash
uvicorn app.main:app --reload
```

This will start the API server, usually accessible at `http://127.0.0.1:8000`. You can then interact with the API endpoints (e.g., `/predict`, `/teams`) using tools like Postman, Insomnia, or `curl`.

## Prediction Algorithm Details

The core of the prediction system is a machine learning model trained on historical EPL match data.

### Data Collection and Preprocessing

*   **Source:** Historical match data is expected to be in `.csv` files, typically one per season, located in `scripts/data/`.
*   **Loading:** The `load_matches` function in `app/services/predictor.py` (and within the Jupyter notebook) reads all CSV files from the `data` directory, concatenates them, and performs initial cleaning.
*   **Cleaning:**
    *   Converts `Date` column to datetime objects, handling multiple formats.
    *   Drops rows with missing values in critical columns (`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`).
    *   Ensures goal columns (`FTHG`, `FTAG`, `HTHG`, `HTAG`) are integer types.

### Feature Engineering

The model relies on engineered features derived from past match performance. For each match, the following rolling averages are calculated for both the home and away teams based on their `window_size` (default 5) previous matches *before* the current match date:

*   **Average Goals Scored (AvgGoalsScored):** The average number of goals the team scored in its last `window_size` matches.
*   **Average Goals Conceded (AvgGoalsConceded):** The average number of goals the team conceded in its last `window_size` matches.
*   **Average Points (AvgPoints):** The average points the team earned in its last `window_size` matches (3 for a win, 1 for a draw, 0 for a loss).

These features provide a measure of a team's recent form and offensive/defensive strength. Additionally, one-hot encoding is applied to `HomeTeam` and `AwayTeam` to include team identity as a feature.

### Model Training and Selection

*   **Models Explored:** The `scripts/EPL WINNER MODEL.ipynb` notebook explores several classification models:
    *   `RandomForestClassifier`
    *   `GradientBoostingClassifier`
    *   `LogisticRegression`
*   **Splitting Data:** Data is split into training (80%) and testing (20%) sets.
*   **Evaluation Metrics:** Models are evaluated using:
    *   Accuracy Score
    *   Classification Report (Precision, Recall, F1-score for each class: Home Win, Draw, Away Win)
    *   Confusion Matrix (visualized)
*   **Selected Model:** The `RandomForestClassifier` was chosen as the primary prediction model due to its robust performance and ability to handle complex relationships in the data.
*   **Feature Importance:** The notebook also visualizes the feature importances from the `RandomForestClassifier`, highlighting which features contribute most to the predictions (e.g., `AwayTeam_AvgPoints`, `HomeTeam_AvgGoalsScored`).

### Prediction Analysis

When a prediction request is made (either via the UI or API):
1.  The `predictor.py` service calculates the rolling average features for the specified home and away teams up to the given `match_date`.
2.  These features, along with one-hot encoded team identities, form the input vector for the trained `RandomForestClassifier`.
3.  The model predicts the most likely outcome (Home Win, Draw, or Away Win) and provides the probability for each outcome.

## API Endpoints

The FastAPI application exposes the following endpoints:

*   **`/` (GET):**
    *   **Description:** Welcome message for the API.
    *   **Response:** `{"message": "Welcome to the EPL Match Predictor API"}`

*   **`/teams` (GET):**
    *   **Description:** Returns a list of all recognized EPL teams that the model can predict for.
    *   **Response:** `{"teams": ["Arsenal", "Chelsea", ...]}`

*   **`/predict` (POST):**
    *   **Description:** Predicts the outcome of an EPL match.
    *   **Request Body (JSON):**
        ```json
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "match_date": "2025-08-01"
        }
        ```
        *   `home_team` (string, required): The name of the home team.
        *   `away_team` (string, required): The name of the away team.
        *   `match_date` (string, optional): The date of the match in 'YYYY-MM-DD' format. If not provided, today's date is used.
    *   **Response (JSON):**
        ```json
        {
            "prediction": "Home Win",
            "probabilities": {
                "Home Win": 0.65,
                "Draw": 0.20,
                "Away Win": 0.15
            }
        }
        ```
        *   `prediction` (string): The predicted outcome ("Home Win", "Draw", or "Away Win").
        *   `probabilities` (object): A dictionary containing the probability for each outcome.
    *   **Error Responses:**
        *   `400 Bad Request`: If `home_team` and `away_team` are the same.
        *   `404 Not Found`: If either `home_team` or `away_team` is not recognized.

import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os

# Define the application
app = FastAPI()

# --- 1. LOAD MODEL AND METADATA ---
MODEL_PATH = os.path.join('models', 'premier_league_match_predictor.pkl')
METADATA_PATH = os.path.join('models', 'model_metadata.pkl')

try:
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    all_teams = metadata['all_teams']
    feature_columns = metadata['feature_columns']
except FileNotFoundError:
    print("Error: Model or metadata files not found. Please run the notebook first.")
    exit()

# --- 2. LOAD HISTORICAL DATA ---
# We need this to calculate team form/rolling stats
DATA_DIR = 'data'
try:
    all_matches = []
    for season_file in os.listdir(DATA_DIR):
        if season_file.endswith('.csv'):
            filepath = os.path.join(DATA_DIR, season_file)
            df = pd.read_csv(filepath, encoding='latin1')
            df['Season'] = season_file.split('.')[0]
            all_matches.append(df)
    
    matches = pd.concat(all_matches, ignore_index=True)
    
    # Basic data cleaning from the notebook
    matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True, errors='coerce')
    matches.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], inplace=True)
    
except FileNotFoundError:
    print(f"Error: Data directory '{DATA_DIR}' not found. Please ensure you have the historical data.")
    exit()

# --- 3. HELPER FUNCTION TO CALCULATE STATS ---
def get_team_stats(df_row, team_name, is_home_team):
    if is_home_team:
        goals_scored = df_row['FTHG']
        goals_conceded = df_row['FTAG']
        result = df_row['FTR']
        points = 3 if result == 'H' else (1 if result == 'D' else 0)
    else:
        goals_scored = df_row['FTAG']
        goals_conceded = df_row['FTHG']
        result = df_row['FTR']
        points = 3 if result == 'A' else (1 if result == 'D' else 0)
    return goals_scored, goals_conceded, points

def get_rolling_averages(team_name, for_date, window_size=5):
    """Calculates the rolling averages for a team before a specific match date."""
    team_matches = matches[
        ((matches['HomeTeam'] == team_name) | (matches['AwayTeam'] == team_name)) &
        (matches['Date'] < for_date)
    ].sort_values(by='Date', ascending=False).head(window_size)

    if len(team_matches) < 1: # Not enough historical data
        return 0, 0, 0

    goals_scored_list, goals_conceded_list, points_list = [], [], []
    
    for _, row in team_matches.iterrows():
        is_home = (row['HomeTeam'] == team_name)
        gs, gc, pts = get_team_stats(row, team_name, is_home)
        goals_scored_list.append(gs)
        goals_conceded_list.append(gc)
        points_list.append(pts)
        
    return (
        sum(goals_scored_list) / len(goals_scored_list),
        sum(goals_conceded_list) / len(goals_conceded_list),
        sum(points_list) / len(points_list)
    )

# --- 4. API ENDPOINTS ---
class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the EPL Match Predictor API"}

@app.get("/teams")
def get_teams():
    """Returns a list of all available teams."""
    return {"teams": all_teams}

@app.post("/predict")
def predict_match(request: MatchPredictionRequest):
    home_team = request.home_team
    away_team = request.away_team

    if home_team not in all_teams or away_team not in all_teams:
        return {"error": "Invalid team name provided."}
    
    if home_team == away_team:
        return {"error": "Home and Away teams cannot be the same."}

    # We'll use today's date for the prediction context
    prediction_date = pd.to_datetime('today')

    # Calculate rolling stats for both teams
    home_avg_gs, home_avg_gc, home_avg_pts = get_rolling_averages(home_team, prediction_date)
    away_avg_gs, away_avg_gc, away_avg_pts = get_rolling_averages(away_team, prediction_date)

    # Create the feature vector for the model
    # Initialize with zeros, just like in the notebook
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Fill in the calculated features
    input_df['HomeTeam_AvgGoalsScored'] = home_avg_gs
    input_df['HomeTeam_AvgGoalsConceded'] = home_avg_gc
    input_df['HomeTeam_AvgPoints'] = home_avg_pts
    input_df['AwayTeam_AvgGoalsScored'] = away_avg_gs
    input_df['AwayTeam_AvgGoalsConceded'] = away_avg_gc
    input_df['AwayTeam_AvgPoints'] = away_avg_pts

    # Fill in the one-hot encoded team columns
    home_col = f'Home_{home_team}'
    away_col = f'Away_{away_team}'
    if home_col in input_df.columns:
        input_df[home_col] = 1
    if away_col in input_df.columns:
        input_df[away_col] = 1
        
    # Ensure the order of columns matches the training order
    input_df = input_df[feature_columns]

    # Make prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Map result to human-readable format
    outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    classes = model.classes_

    response = {
        "prediction": outcome_map.get(prediction, "Unknown"),
        "probabilities": {
            outcome_map.get(cls): prob for cls, prob in zip(classes, probabilities)
        }
    }

    return response

import os
import pandas as pd
import joblib
from app.config import MODEL_PATH, METADATA_PATH, DATA_DIR
from app.utils.stats import get_rolling_averages
import time

# Load model and metadata
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
all_teams = metadata['all_teams']
feature_columns = metadata['feature_columns']

# Load historical match data
def load_matches():
    all_matches = []
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_DIR, file), encoding='latin1')
            df['Season'] = file.split('.')[0]
            all_matches.append(df)
    matches = pd.concat(all_matches, ignore_index=True)
    matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True, errors='coerce')
    matches.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], inplace=True)
    return matches

matches = load_matches()

def predict(home_team: str, away_team: str, match_date: pd.Timestamp):
    home_avg_gs, home_avg_gc, home_avg_pts = get_rolling_averages(matches, home_team, match_date)
    away_avg_gs, away_avg_gc, away_avg_pts = get_rolling_averages(matches, away_team, match_date)

    # Create the feature vector for the model
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Fill in the calculated features
    input_df['HomeTeam_AvgGoalsScored'] = home_avg_gs
    input_df['HomeTeam_AvgGoalsConceded'] = home_avg_gc
    input_df['HomeTeam_AvgPoints'] = home_avg_pts
    input_df['AwayTeam_AvgGoalsScored'] = away_avg_gs
    input_df['AwayTeam_AvgGoalsConceded'] = away_avg_gc
    input_df['AwayTeam_AvgPoints'] = away_avg_pts

    # One-hot encode team
    home_col = f'Home_{home_team}'
    away_col = f'Away_{away_team}'
    if home_col in input_df.columns:
        input_df[home_col] = 1
    if away_col in input_df.columns:
        input_df[away_col] = 1

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    classes = model.classes_
    
    time.sleep(10)
    return {
        "prediction": outcome_map.get(prediction, "Unknown"),
        "probabilities": {outcome_map.get(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    }

from fastapi import APIRouter, HTTPException
from app.models.schemas import MatchPredictionRequest
from app.services.predictor import predict, all_teams
import pandas as pd

router = APIRouter()



@router.post("/predict")
def predict_result(payload: MatchPredictionRequest):
    home = payload.home_team.strip().title()
    away = payload.away_team.strip().title()
    match_date = pd.to_datetime(payload.match_date or 'today')

    if home == away:
        raise HTTPException(status_code=400, detail="Home and Away teams cannot be the same.")
    if home not in all_teams or away not in all_teams:
        raise HTTPException(status_code=404, detail="Team not recognized.")

    return predict(home, away, match_date)

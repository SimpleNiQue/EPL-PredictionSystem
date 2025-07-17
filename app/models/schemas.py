from pydantic import BaseModel, Field
from typing import Optional

class MatchPredictionRequest(BaseModel):
    home_team: str = Field(..., example="Arsenal")
    away_team: str = Field(..., example="Chelsea")
    match_date: Optional[str] = Field(None, example="2025-08-01")

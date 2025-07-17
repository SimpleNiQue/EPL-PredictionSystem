import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, '..', 'scripts/models', 'premier_league_match_predictor.pkl')
METADATA_PATH = os.path.join(BASE_DIR, '..', 'scripts/models', 'model_metadata.pkl')
DATA_DIR = os.path.join(BASE_DIR, '..', 'scripts/data')

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routes.endpoints import router
from app.services.predictor import all_teams

app = FastAPI(title="EPL Match Predictor")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Include API router
app.include_router(router)

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "teams": all_teams})

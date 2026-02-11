import os
import io
import base64
import uvicorn
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from src.model_explaining import SHAPExplanationStrategy

# 1. UNE SEULE CRÉATION D'APP
app = FastAPI()

# 2. CONFIGURATION DES CHEMINS
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/app/artifacts")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(ARTIFACTS_DIR, "model.joblib"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
ENCODERS_PATH = os.getenv("ENCODERS_PATH", os.path.join(ARTIFACTS_DIR, "encoders.joblib"))

MODEL = None
SCALER = None
ENCODERS = None
shap_strategy = SHAPExplanationStrategy()

def load_artifacts_from_disk():
    global MODEL, SCALER, ENCODERS
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        SCALER = joblib.load(SCALER_PATH)
    if os.path.exists(ENCODERS_PATH):
        ENCODERS = joblib.load(ENCODERS_PATH)
    print("✅ Artefacts chargés")

# 3. UN SEUL BLOC STARTUP
@app.on_event("startup")
async def startup():
    # CETTE LIGNE EST ESSENTIELLE POUR /metrics
    Instrumentator().instrument(app).expose(app)
    load_artifacts_from_disk()

class StudentInput(BaseModel):
    age: float
    education_level: str
    study_hours_per_day: float
    uses_ai: str
    ai_tools_used: str
    purpose_of_ai: str
    grades_before_ai: float
    grades_after_ai: float
    daily_screen_time_hours: float

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: StudentInput):
    try:
        if MODEL is None: raise RuntimeError("Model not loaded")
        df = pd.DataFrame([data.dict()])
        # (Tes transformations ici...)
        X_final = df[MODEL.feature_names_in_]
        prob = float(MODEL.predict_proba(X_final)[0][1])
        return {"risk": int(prob > 0.5), "probability": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
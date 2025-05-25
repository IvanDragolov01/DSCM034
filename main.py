from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import pickle
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Prediction API")

try:
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    logger.error("Model not found!")
    raise RuntimeError("Model file is missing.")

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_items=4, max_items=4)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Internal error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})

@app.get("/health")
def health_check():
    return {"status": "alive"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        input_data = np.array(request.features).reshape(1, -1)
        if not np.all(np.isfinite(input_data)):
            raise HTTPException(status_code=400, detail="Input contains non-numeric values.")
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")

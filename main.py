from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# =================================================================
# INSTRUCTIONS FOR SWITCHING MODELS:
# 1. Update 'MODEL_FILE' and 'COLUMNS_FILE' constants below.
# 2. Update the 'PredictionInput' class schema to match your 
#    new model's required input features.
# =================================================================

# 1. Configuration: Model file settings
MODEL_FILE = 'weather_model.joblib'
COLUMNS_FILE = 'model_columns.joblib'

app = FastAPI()

# 2. CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model Ingestion: Load serialized model and columns
try:
    model = joblib.load(MODEL_FILE)
    model_columns = joblib.load(COLUMNS_FILE)
    print(f"SUCCESS: Loaded {MODEL_FILE} and {COLUMNS_FILE}")
except Exception as e:
    print(f"ERROR: Failed to load model files ({MODEL_FILE}, {COLUMNS_FILE}): {e}")

# 4. Generic Request Schema
class PredictionInput(BaseModel):
    # Features for time-series prediction
    # Update these fields when switching to a different model
    month: int
    lag1: float
    lag2: float
    lag3: float

@app.get("/")
def read_root():
    # Generic health check endpoint
    return {"status": "Prediction API is active", "model": MODEL_FILE}

@app.post("/predict")
def predict(data: PredictionInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Handle categorical variables (kept for compatibility)
    input_df = pd.get_dummies(input_df)
    
    # Align DataFrame columns with the trained model's features
    final_df = pd.DataFrame(columns=model_columns)
    final_df = pd.concat([final_df, input_df]).fillna(0)
    final_df = final_df[model_columns]
    
    # Execute prediction
    prediction = model.predict(final_df)[0]
    
    # Return result with a generic key
    return {"prediction_result": float(prediction)}

# Entry point for Cloud Run
if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable or default to 8080
    port = int(os.environ.get("PORT", 8080))
    # host="0.0.0.0" is required for Cloud Run
    uvicorn.run(app, host="0.0.0.0", port=port)
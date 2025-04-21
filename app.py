from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing objects
model = joblib.load("artifacts/model.pkl")
label_encoders = joblib.load("artifacts/label_encoders.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
target_encoder = joblib.load("artifacts/target_encoder.pkl")

app = FastAPI(
    title="Easy Labor Prediction API",
    description="Predicts US labor case status (Certified or Denied) based on applicant & job details.",
    version="1.0.0"
)

# ✅ Map U.S. states to employment regions
state_to_region_map = {
    "California": "West",
    "Nevada": "West",
    "Texas": "South",
    "Florida": "South",
    "New York": "Northeast",
    "Massachusetts": "Northeast",
    "Illinois": "Midwest",
    "Ohio": "Midwest",
    "Puerto Rico": "Island",
    "Hawaii": "Island"
    # ➕ Add more if needed
}

# Input schema
class InputData(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    region_of_employment: str  # Can be a region or a U.S. state
    unit_of_wage: str
    full_time_position: str
    no_of_employees: float
    yr_of_estab: float
    prevailing_wage: float

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Easy Labor Case Status Prediction API"}

@app.post("/predict", tags=["Prediction"])
def predict(data: InputData):
    try:
        input_dict = data.dict()

        # Handle region/state conversion
        region = input_dict["region_of_employment"]
        if region in state_to_region_map:
            input_dict["region_of_employment"] = state_to_region_map[region]

        # Categorical columns
        cat_features = [
            'continent', 'education_of_employee', 'has_job_experience',
            'requires_job_training', 'region_of_employment',
            'unit_of_wage', 'full_time_position'
        ]

        # Encode categorical values 
        for col in cat_features:
            value = input_dict[col]
            encoder = label_encoders.get(col)

            if value not in encoder.classes_:
                return {"error": f"Invalid input '{value}' for '{col}'. Expected one of: {list(encoder.classes_)}"}

            input_dict[col] = int(encoder.transform([value])[0])

        # Numerical values (scaled)
        num_features = ['no_of_employees', 'yr_of_estab', 'prevailing_wage']
        scaled_values = scaler.transform([[input_dict[col] for col in num_features]])[0]

        #  Final input for model
        final_input = np.array([
            input_dict['continent'],
            input_dict['education_of_employee'],
            input_dict['has_job_experience'],
            input_dict['requires_job_training'],
            input_dict['region_of_employment'],
            input_dict['unit_of_wage'],
            input_dict['full_time_position'],
            *scaled_values
        ]).reshape(1, -1)

        # Model prediction
        prediction = model.predict(final_input)[0]

        # Decode prediction if needed
        prediction_label = target_encoder.inverse_transform([prediction])[0]

        return {
            "prediction_raw": int(prediction),
            "prediction_label": prediction_label
        }

    except Exception as e:
        return {"error": str(e)}

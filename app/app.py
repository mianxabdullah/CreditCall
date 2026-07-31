from fastapi import FastAPI
import joblib

app = FastAPI(title="Loan Approval API")

model = joblib.load('models/loan_model.pkl')
scaler = joblib.load('models/scaler.pkl')
model_columns = joblib.load('models/model_columns.pkl')

@app.get("/")
def home():
    return {"message": "Welcome to the Loan Approval API!",
            "status": "Active",
            "version": "1.0.0",
            "endpoints": "/health, /predict, /predict_file, /applicants, /applicants/{id} (GET/PUT/DELETE), /predictions, /predictions/{applicant_id}"
            }


@app.get("/health")
def health_check():
    return {"status": "Healthy",
            "version": "1.0.0",}
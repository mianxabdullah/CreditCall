from fastapi import FastAPI, HTTPException, Header,Depends
from app.schemas import LoanApplication 
import pandas as pd
from app.preprocess import preprocess 
from app.db import save_applicant, save_prediction
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(...)): # Added authentication to the API using an API key. The function verify_api_key checks if the provided x_api_key header matches the expected API_KEY. If it doesn't match, it raises an HTTPException with a 401 status code, indicating unauthorized access.
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

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



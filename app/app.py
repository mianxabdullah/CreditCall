from fastapi import FastAPI, HTTPException, Header,Depends
from app.schemas import LoanApplication 
import pandas as pd
from app.preprocess import preprocess 
from app.db import save_applicant, save_prediction, get_all_applicants
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

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(application: LoanApplication):
    try:
        input_dict = application.model_dump()
        df = pd.DataFrame([input_dict])

        scaled_input, _ = preprocess(df, scaler, model_columns)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        applicant_id = save_applicant({
            "gender": application.Gender,
            "married": application.Married,
            "dependents": application.Dependents,
            "education": application.Education,
            "self_employed": application.Self_Employed,
            "applicant_income": application.ApplicantIncome,
            "coapplicant_income": application.CoapplicantIncome,
            "loan_amount": application.LoanAmount,
            "loan_amount_term": application.Loan_Amount_Term,
            "credit_history": application.Credit_History,
            "property_area": application.Property_Area,
        })

        loan_status = "Approved" if prediction == 1 else "Rejected"
        save_prediction(applicant_id, loan_status, round(float(probability), 4))

        return {
            "loan_status": loan_status,
            "approval_probability": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/applicants", dependencies=[Depends(verify_api_key)])
def list_applicants():
    return get_all_applicants()



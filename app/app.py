from fastapi import FastAPI, HTTPException, Header,Depends,UploadFile, File
from app.schemas import LoanApplication , ApplicantUpdate
import pandas as pd
from app.preprocess import preprocess 
from app.db import (save_applicant, save_prediction, get_all_applicants,get_applicant_by_id, update_applicant,
                    delete_applicant,get_all_predictions,get_predictions_by_applicant)
import joblib
import os
import io
from fastapi.responses import StreamingResponse
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


@app.get("/applicants/{applicant_id}", dependencies=[Depends(verify_api_key)])
def read_applicant(applicant_id: int):
    applicant = get_applicant_by_id(applicant_id)
    if applicant is None:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return applicant


@app.put("/applicants/{applicant_id}", dependencies=[Depends(verify_api_key)])
def edit_applicant(applicant_id: int, applicant: ApplicantUpdate):
    # Convert the Pydantic model to a dictionary and prepare the data for the update operation
    data = {
        "gender": applicant.Gender,
        "married": applicant.Married,
        "dependents": applicant.Dependents,
        "education": applicant.Education,
        "self_employed": applicant.Self_Employed,
        "applicant_income": applicant.ApplicantIncome,
        "coapplicant_income": applicant.CoapplicantIncome,
        "loan_amount": applicant.LoanAmount,
        "loan_amount_term": applicant.Loan_Amount_Term,
        "credit_history": applicant.Credit_History,
        "property_area": applicant.Property_Area,
    }
    updated = update_applicant(applicant_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return get_applicant_by_id(applicant_id)


@app.delete("/applicants/{applicant_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def remove_applicant(applicant_id: int):
    deleted = delete_applicant(applicant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Applicant not found")

@app.get("/predictions", dependencies=[Depends(verify_api_key)])
def list_predictions():
    return get_all_predictions()

@app.get("/predictions/{applicant_id}", dependencies=[Depends(verify_api_key)])
def read_predictions_for_applicant(applicant_id: int):
    applicant = get_applicant_by_id(applicant_id)
    if applicant is None:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return get_predictions_by_applicant(applicant_id)


@app.post("/predict_file", dependencies=[Depends(verify_api_key)])
async def predict_loan_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    try:
        df = pd.read_csv(file.file)

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty.")

        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)

        for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
            df[col] = df[col].fillna(df[col].mode()[0])

        df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

        raw_rows = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                        'Loan_Amount_Term', 'Credit_History', 'Property_Area']].copy()

        scaled, missing_features = preprocess(df, scaler, model_columns)

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={"message": "CSV is missing required columns", "missing_features": missing_features}
            )

        predictions = model.predict(scaled)
        probabilities = model.predict_proba(scaled)[:, 1]

        



    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")




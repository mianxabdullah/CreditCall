import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ALL_FIELDS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

FIELD_LABELS = {
    "Gender": "gender",
    "Married": "marital status",
    "Dependents": "number of dependents",
    "Education": "education (Graduate/Not Graduate)",
    "Self_Employed": "whether you're self-employed",
    "ApplicantIncome": "your monthly income",
    "CoapplicantIncome": "co-applicant's monthly income (say 0 if none)",
    "LoanAmount": "the loan amount you're requesting",
    "Loan_Amount_Term": "the loan term in days",
    "Credit_History": "your credit history (any missed/defaulted payments?)",
    "Property_Area": "property area (Urban, Semiurban, or Rural)"
}



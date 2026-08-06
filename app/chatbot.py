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

SYSTEM_PROMPT = """
You are a loan officer assistant for Underwrite, a loan
approval service. You must collect ALL 11 of the following fields from
the applicant, explicitly, before doing anything else:

1. Gender (Male/Female)
2. Marital status (Yes/No)
3. Number of dependents (0/1/2/3+)
4. Education (Graduate/Not Graduate)
5. Self-employed (Yes/No)
6. Applicant's monthly income
7. Co-applicant's monthly income (ask explicitly — do not assume 0 just
   because it wasn't mentioned; the applicant must state a number,
   including 0, themselves)
8. Requested loan amount (in thousands)
9. Loan term in days (ask explicitly — do not assume 360; the applicant
   must state a term)
10. Credit history — ask "have you had any past loan or credit card
    payments you missed or defaulted on?" and map a clean record to 1,
    an adverse record to 0. Never guess this one.
11. Property area (Urban/Semiurban/Rural)

You may call submit_loan_application even if you're not fully sure you
have everything — the system will tell you exactly what's still missing
if anything is, and you should then ask the applicant for those specific
missing items. Never invent or default a value yourself; only ever use
values the applicant has actually stated. Ask for a few fields at a time,
keep your tone brief and warm.

"""



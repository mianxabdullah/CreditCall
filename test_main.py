from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)
API_KEY = os.getenv("API_KEY")

VALID_APPLICANT = {
    "Gender": "Male", "Married": "Yes", "Dependents": "0",
    "Education": "Graduate", "Self_Employed": "No",
    "ApplicantIncome": 5000, "CoapplicantIncome": 0,
    "LoanAmount": 150, "Loan_Amount_Term": 360,
    "Credit_History": 1, "Property_Area": "Urban"
}


def test_health_check():
    """The health endpoint should always work, no auth needed."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Healthy"


def test_predict_without_api_key_header_is_rejected():
    """No X-API-Key header at all -> FastAPI's own request validation
    rejects it before our code runs (422), since Header(...) is required."""
    response = client.post("/predict", json=VALID_APPLICANT)
    assert response.status_code == 422


def test_predict_with_wrong_api_key_is_rejected():
    """Header present but value doesn't match -> our verify_api_key
    check rejects it (401)."""
    response = client.post(
        "/predict",
        headers={"x-api-key": "wrong-key-entirely"},
        json=VALID_APPLICANT
    )
    assert response.status_code == 401



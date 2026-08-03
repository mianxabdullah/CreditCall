from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL) # Create a database engine that means we can connect to the database and execute SQL queries. The engine is created using the connection string stored in the DATABASE_URL environment variable.

def save_applicant(data: dict) -> int: # -> int: means this function returns an integer, which is the new applicant's ID
    """Inserts an applicant row, returns the new id."""
    query = text(""" 
        INSERT INTO applicants (gender, married, dependents, education, self_employed,
                                 applicant_income, coapplicant_income, loan_amount,
                                 loan_amount_term, credit_history, property_area)
        VALUES (:gender, :married, :dependents, :education, :self_employed,
                :applicant_income, :coapplicant_income, :loan_amount,
                :loan_amount_term, :credit_history, :property_area)
        RETURNING id
    """)
    #text() is used to create a SQL statement that can be executed against the database.
    #  The placeholders (e.g., :gender, :married) are used to safely pass parameters to the SQL query, 
    #preventing SQL injection attacks. The RETURNING id clause allows us to get the ID of the newly inserted applicant.
    with engine.connect() as conn: # engine.connect() establishes a connection to the database. The with statement ensures that the connection is properly closed after the block of code is executed, even if an error occurs.
        result = conn.execute(query, data) # Execute the query with the provided data. it doesnt automatically commit the transaction, so we need to call conn.commit() to save the changes to the database. The result object contains the result of the query execution, which we can use to retrieve the newly inserted applicant's ID.
        conn.commit() # Commit the transaction to save the changes to the database. This is important because, without committing, the changes would not be persisted in the database.
        return result.scalar_one() # scalar_one() retrieves the first column of the first row from the result set, which in this case is the ID of the newly inserted applicant. If no rows are returned, it raises an exception. This ensures that we get a single integer value representing the new applicant's ID.


def save_prediction(applicant_id: int, loan_status: str, probability: float):
    """Inserts a prediction row linked to an applicant."""
    query = text("""
        INSERT INTO predictions (applicant_id, loan_status, approval_probability)
        VALUES (:applicant_id, :loan_status, :approval_probability)
    """)
    with engine.connect() as conn:
        conn.execute(query, {
            "applicant_id": applicant_id,
            "loan_status": loan_status,
            "approval_probability": probability
        })
        conn.commit()


def get_all_applicants():
    query = text("SELECT * FROM applicants ORDER BY created_at DESC")
    with engine.connect() as conn:
        result = conn.execute(query)
        return [dict(row._mapping) for row in result] # The list comprehension iterates over each row in the result set, converting each row into a dictionary using row._mapping. This allows us to return a list of dictionaries, where each dictionary represents an applicant's data, making it easier to work with in the API response.


def get_applicant_by_id(applicant_id: int):
    query = text("SELECT * FROM applicants WHERE id = :id")
    with engine.connect() as conn:
        result = conn.execute(query, {"id": applicant_id})
        row = result.mappings().first() # The result.mappings() method returns an iterable of dictionaries, where each dictionary represents a row in the result set. The first() method retrieves the first row from the result set, or None if there are no rows. This allows us to get the applicant's data as a dictionary if it exists, or None if the applicant with the specified ID does not exist in the database.
        return dict(row) if row else None

def update_applicant(applicant_id: int, data: dict) -> bool:
    query = text("""
        UPDATE applicants
        SET gender = :gender, married = :married, dependents = :dependents,
            education = :education, self_employed = :self_employed,
            applicant_income = :applicant_income, coapplicant_income = :coapplicant_income,
            loan_amount = :loan_amount, loan_amount_term = :loan_amount_term,
            credit_history = :credit_history, property_area = :property_area
        WHERE id = :id
    """)
    data["id"] = applicant_id
    with engine.connect() as conn:
        result = conn.execute(query, data)
        conn.commit()
        return result.rowcount > 0

def delete_applicant(applicant_id: int) -> bool:
    query = text("DELETE FROM applicants WHERE id = :id")
    with engine.connect() as conn:
        result = conn.execute(query, {"id": applicant_id})
        conn.commit()
        return result.rowcount > 0
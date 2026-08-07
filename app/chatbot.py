import os
from dotenv import load_dotenv
from groq import Groq
from app.db import get_chat_history, delete_chat_history, save_chat_history
import json 

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

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_loan_application",
        "description": "Submit the loan application with whatever fields have been collected so far. It's fine to call this even if some fields are still missing — the system will tell you which ones and you can continue the conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "Gender": {"type": "string", "enum": ["Male", "Female"]},
                "Married": {"type": "string", "enum": ["Yes", "No"]},
                "Dependents": {"type": "string", "enum": ["0", "1", "2", "3+"]},
                "Education": {"type": "string", "enum": ["Graduate", "Not Graduate"]},
                "Self_Employed": {"type": "string", "enum": ["Yes", "No"]},
                "ApplicantIncome": {"type": "number"},
                "CoapplicantIncome": {"type": "number"},
                "LoanAmount": {"type": "number"},
                "Loan_Amount_Term": {"type": "number"},
                "Credit_History": {"type": "integer", "enum": [0, 1]},
                "Property_Area": {"type": "string", "enum": ["Urban", "Semiurban", "Rural"]}
            }
            # Deliberately no "required" list here requiring all fields causes a hard 400 error the moment the model calls the tool with anything missing 
        }
    }
}

def get_or_create_conversation(session_id: str) -> list[dict]:
    history = get_chat_history(session_id)
    if history is None:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
    return history

def reset_conversation(session_id: str):
    delete_chat_history(session_id)    


def chat_turn(session_id: str, user_message: str, _depth: int = 0):
    """
    Sends the user's message to Groq, manages the complete tool-calling workflow 
    (including prompting for any missing loan application details), and saves 
    conversation history to Postgres after every turn so chats persist across restarts 
    and multiple server instances.    

    Returns:
      - {"type": "message", "text": "..."}            -> show this, keep chatting
      - {"type": "ready_to_submit", "fields": {...}}    -> every field present
    """
    if _depth > 4:
        return {"type": "message", "text": "Sorry, I'm having trouble processing that — could you try rephrasing?"}

    history = get_or_create_conversation(session_id)
    if _depth == 0:
        history.append({"role": "user", "content": user_message}) # Add the user's message to the conversation history. This is done only on the initial call (depth 0) to avoid duplicating the user's message in recursive calls when the model asks for missing fields.

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            tools=[SUBMIT_TOOL], # The model is provided with the submit_loan_application tool, which allows it to submit the loan application with the fields it has collected so far. The model can call this tool even if some fields are still missing, and the system will inform it of any missing fields so it can continue the conversation to collect them.
            tool_choice="auto", # The model will automatically decide when to call the submit_loan_application tool based on the conversation context and the fields it has collected so far. This allows for a more natural interaction, as the model can determine the appropriate time to submit the application without requiring explicit instructions from the user.
            temperature=0, # The temperature is set to 0 to make the model's responses more deterministic and focused, reducing randomness in its output. This is important for a loan application assistant, as we want consistent and reliable responses when collecting sensitive information from applicants.
            max_tokens=1024 # The maximum number of tokens (words or word pieces) that the model can generate in its response is set to 1024. This ensures that the model has enough capacity to provide detailed responses and ask for any missing information without being cut off prematurely.
        )
    except Exception as e:
        return {"type": "message", "text": f"(Assistant is having trouble responding right now: {str(e)})"}

    message = response.choices[0].message # Response from LLM

    # Build the assistant's history entry manually rather than using model_dump() to avoid including the tool_call
    # field when its not called because groq-python will throw an error when trying to serialize the 
    # tool_call field as None. So, we only include the tool_calls field in the assistant's history entry if there are 
    # actual tool calls present in the message.

    assistant_entry = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        assistant_entry["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            }
            for tc in message.tool_calls
        ]
    history.append(assistant_entry)

    if not message.tool_calls:
        save_chat_history(session_id, history)
        return {"type": "message", "text": message.content or ""}

    
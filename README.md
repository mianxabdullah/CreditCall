# CreditCall

A REST API that predicts loan approval decisions using a machine learning
model trained on historical loan data, backed by a PostgreSQL database
that logs every applicant and prediction made. Includes a web form, an
admin panel, and a conversational chatbot — three different front doors
into the same prediction pipeline.

## What it does

Send an applicant's details (income, credit history, employment status,
etc.) to the API — via a form, a CSV upload, or a natural-language chat —
and it returns an approval decision along with a confidence score. Every
applicant and prediction gets saved to the database, so you can search
and review prediction history over time through the admin panel.

## Tech stack

- **FastAPI** — the web framework serving the API
- **scikit-learn** — trains and runs the ML model (Logistic Regression)
- **PostgreSQL** — stores applicants, prediction history, and chatbot
  conversation state
- **SQLAlchemy** (raw SQL, not ORM) — talks to the database
- **pandas / joblib** — data preprocessing and model loading
- **Groq (Llama 3.3 70B) with tool calling** — powers the conversational
  chatbot, extracting structured application data from natural language
- **pytest** — automated test suite covering auth, validation, and errors
- Plain HTML/CSS/JS frontends (no framework) — a "ledger/case file"
  visual theme shared across all three interfaces

## The ML model

Trained on the Kaggle "Loan Prediction Problem Dataset" (~614 rows).
After comparing plain Logistic Regression, Random Forest, and a
class-weighted Logistic Regression, the weighted model was chosen —
it reduces false approvals (predicting "approved" for applicants who
should have been rejected) at a small cost to overall accuracy, which
is the more realistic priority for a lending decision.

| Model | Accuracy | Rejected Recall | False Approvals |
|---|---|---|---|
| Logistic Regression | 86.2% | 58% | 16 |
| Random Forest | 80.5% | 58% | 16 |
| **Weighted Logistic Regression (used)** | 82.1% | **68%** | **12** |

## Project structure

```
creditcall/
├── data/                     # raw training data (CSV)
├── models/                   # trained model, scaler, and column list (.pkl)
├── notebooks/                # EDA + model training notebook
├── app/
│   ├── main.py                 # FastAPI app, all routes, API key auth
│   ├── db.py                    # database queries (raw SQL) — applicants,
│   │                               predictions, and chatbot session storage
│   ├── schemas.py                 # request/response validation
│   ├── preprocess.py                # shared encoding/scaling logic, used
│   │                                   by /predict, /predict_file, /chat
│   └── chatbot.py                     # Groq client, tool-calling loop,
│                                         conversational field extraction
├── static/
│   ├── index.html               # public application form + batch CSV upload
│   ├── login.html                 # admin sign-in (API key entry)
│   ├── history.html                 # admin panel: search/view/edit/delete
│   └── chat.html                      # conversational application assistant
├── test_main.py               # pytest suite
├── requirements.txt
└── .env                         # DATABASE_URL, API_KEY, GROQ_API_KEY (not committed)
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a PostgreSQL database, then create a `.env` file in the
   project root:
   ```
   DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/creditcall_db
   API_KEY=choose-any-secret-string-here
   GROQ_API_KEY=your-groq-api-key
   ```

3. Create the tables (run in pgAdmin's Query Tool or `psql`):
   ```sql
   CREATE TABLE applicants (
       id SERIAL PRIMARY KEY,
       gender VARCHAR(10) NOT NULL,
       married VARCHAR(3) NOT NULL,
       dependents VARCHAR(2) NOT NULL,
       education VARCHAR(20) NOT NULL,
       self_employed VARCHAR(3) NOT NULL,
       applicant_income FLOAT NOT NULL,
       coapplicant_income FLOAT NOT NULL,
       loan_amount FLOAT NOT NULL,
       loan_amount_term FLOAT NOT NULL,
       credit_history INTEGER NOT NULL,
       property_area VARCHAR(20) NOT NULL,
       created_at TIMESTAMPTZ DEFAULT NOW()
   );

   CREATE TABLE predictions (
       id SERIAL PRIMARY KEY,
       applicant_id INTEGER NOT NULL REFERENCES applicants(id) ON DELETE CASCADE,
       loan_status VARCHAR(10) NOT NULL,
       approval_probability FLOAT NOT NULL,
       created_at TIMESTAMPTZ DEFAULT NOW()
   );

   CREATE TABLE chat_sessions (
       session_id VARCHAR(100) PRIMARY KEY,
       history JSONB NOT NULL DEFAULT '[]',
       updated_at TIMESTAMPTZ DEFAULT NOW()
   );
   ```

4. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open `http://127.0.0.1:8000/docs` for interactive API documentation,
   or go straight to the frontends:
   - `http://127.0.0.1:8000/static/index.html` — application form
   - `http://127.0.0.1:8000/static/login.html` — admin panel sign-in
   - `http://127.0.0.1:8000/static/chat.html` — chat with an assistant

   Note: the frontends have a placeholder API key baked into their
   `<script>` section (`const API_KEY = "..."`) since this is a
   client-side demo, not a production auth setup — see "Known
   simplifications" below. Replace it with the value from your `.env`.

## Running tests

```bash
pytest test_main.py -v
```

Covers: health check, auth rejection (missing/wrong API key), valid
prediction requests, input validation errors, and 404 handling.

## Endpoints

| Method | Endpoint | Auth required | Description |
|---|---|---|---|
| GET | `/health` | No | Health check |
| POST | `/predict` | Yes | Submit one applicant, get an approval decision |
| POST | `/predict_file` | Yes | Upload a CSV of applicants, get back a CSV with a `Status` column added — each row is also saved to the database |
| POST | `/chat` | Yes | Send a chat message; the assistant asks follow-up questions until it has every field, then returns a prediction |
| GET | `/applicants` | Yes | List all applicants |
| GET | `/applicants/{id}` | Yes | Get one applicant |
| PUT | `/applicants/{id}` | Yes | Update an applicant's info |
| DELETE | `/applicants/{id}` | Yes | Delete an applicant (cascades to their predictions) |
| GET | `/predictions` | Yes | List all predictions, with applicant details joined in |
| GET | `/predictions/{applicant_id}` | Yes | Get prediction history for one applicant |

Protected endpoints require an `x-api-key` header matching `API_KEY` in `.env`.

## The three frontends

- **Application form** (`index.html`) — the primary loan application
  experience: a form styled as a case dossier, plus a batch CSV upload
  section. Approval/rejection is shown as an animated ink-stamp verdict.
- **Admin panel** (`login.html` → `history.html`) — sign in with the API
  key, then search for a specific case by ID or browse all records.
  Supports viewing full applicant detail + prediction history, editing,
  and deleting.
- **Chat assistant** (`chat.html`) — a conversational alternative to the
  form. Powered by Groq's Llama 3.3 70B with tool calling: the model
  collects the same 11 required fields through natural conversation,
  asking follow-up questions for anything missing, then submits through
  the exact same prediction pipeline as the form.

## Notes on design decisions

- **Raw SQL over an ORM** — chosen deliberately while learning, to
  understand exactly what queries run against the database rather than
  relying on an abstraction layer.
- **Cascade delete on predictions** — deleting an applicant also removes
  their prediction history, keeping the two tables consistent.
- **Shared `preprocess()` function** — `/predict`, `/predict_file`, and
  `/chat` all run identical encoding and scaling steps. Instead of
  duplicating that logic in each endpoint, it lives in one place
  (`app/preprocess.py`), so a future change only needs to happen once.
- **Chatbot field completeness is checked server-side, not trusted from
  the LLM** — the tool schema intentionally has no `required` list
  (Groq validates tool-call arguments against the schema before the
  response reaches application code, so requiring all fields caused
  hard 400 errors the moment anything was missing). Instead, the
  backend checks which of the 11 fields are present after every tool
  call and, if any are missing, feeds that back to the model so it can
  naturally ask for them — a self-correcting loop rather than a single
  one-shot extraction.
- **Chat history is stored in Postgres (`chat_sessions`), not in
  memory** — so conversations survive a server restart, unlike a
  simple in-memory dictionary would.
- **Every chatbot submission is re-validated through the same
  `LoanApplication` schema `/predict` uses** — even though the model is
  instructed carefully, LLM output isn't trusted blindly before it
  reaches the ML pipeline.

## Known simplifications

- The frontend API key is hardcoded in each HTML file's `<script>`
  section rather than kept server-side. Since it's visible to anyone
  who opens browser dev tools, this is a reasonable simplification for
  a learning/demo project but not how a production app would handle
  client authentication.
- Admin "login" is a single shared API key, not individual user
  accounts — there's no per-user identity, just one credential that
  grants admin access.
- The chatbot's conversation memory is per `session_id` with no
  expiry — long-abandoned sessions will accumulate in `chat_sessions`
  unless cleaned up manually.
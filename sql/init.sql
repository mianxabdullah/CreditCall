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

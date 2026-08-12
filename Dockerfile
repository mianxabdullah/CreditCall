FROM python:3.13-slim

WORKDIR /code

# Copy just requirements first and install — Docker caches this layer,
# so rebuilding after a code-only change won't reinstall every package,
# only when requirements.txt itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project (app code, models, static frontends)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

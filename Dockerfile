# Use a temporary builder image
FROM python:3.9-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Use a final lightweight image
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .



ENV PORT=8080
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/gcs-key.json"
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
    
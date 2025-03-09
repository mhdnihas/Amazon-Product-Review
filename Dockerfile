# Use a smaller base image
FROM python:3.9-slim as base

# Create a virtual environment and install dependencies
FROM base as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Final stage
FROM base
WORKDIR /app
COPY . .

EXPOSE 8000

ENV PATH="/opt/venv/bin:$PATH"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
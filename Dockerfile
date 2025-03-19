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
COPY Models/ /app/Models/

# Expose the port (Render uses port 10000 by default)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["sh", "-c", "gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT"]

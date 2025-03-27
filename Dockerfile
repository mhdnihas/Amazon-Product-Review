# Use an official lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    NLTK_DATA=/app/nltk_data

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY app.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN mkdir -p /app/nltk_data && \
    python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); \
                      nltk.download('wordnet', download_dir='/app/nltk_data'); \
                      nltk.download('averaged_perceptron_tagger', download_dir='/app/nltk_data'); \
                      nltk.download('maxent_ne_chunker', download_dir='/app/nltk_data'); \
                      nltk.download('punkt_tab', download_dir='/app/nltk_data')"

# Expose the port
EXPOSE 8080

# Run the application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

import os
import string
import pickle
import numpy as np
import uvicorn
import nltk
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import json
from google.cloud import storage
import csv
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

BUCKET_NAME = "amazon-product-review"
FILE_NAME = "Current_User_Reviews.csv"

# Setup for different environments
IS_LOCAL = os.environ.get('GAE_ENV', '').startswith('standard')
if not IS_LOCAL:
    logger.info("Running in Google App Engine environment - using default credentials")
else:
    # Local development with explicit credentials
    if os.path.exists("amazonproductreviewsentiment-eb289026e366.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "amazonproductreviewsentiment-eb289026e366.json"
        logger.info("Using local credentials file")
    elif os.path.exists("/app/gcs-key.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/gcs-key.json"
        logger.info("Using deployed credentials file")
    else:
        logger.warning("WARNING: No credentials file found, using environment variable")

app = FastAPI()

# Load the LSTM model
with open("Models/lstm_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
model.load_weights("Models/lstm_model.weights.h5")

# Load tokenizer
with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    """Preprocesses input text by applying lowercasing, tokenization, punctuation removal,
    stopword removal, numeric filtering, and stemming."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if not word.isdigit()]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Store feedback in Google Cloud Storage
def store_feedback(review, predicted_sentiment, actual_sentiment=None, rating=None):
    """Stores user feedback in a CSV file on GCS."""
    feedback_data = [review, predicted_sentiment, actual_sentiment, rating]
    return upload_csv_to_gcs(feedback_data)

def upload_csv_to_gcs(data):
    """Appends feedback data to an existing CSV file in GCS, ensuring no duplicates."""
    try:
        # Use default credentials which will work in App Engine environment
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(FILE_NAME)

        header = ["review", "predicted_sentiment", "actual_sentiment", "rating"]

        existing_data = []
        if blob.exists():
            try:
                existing_file = blob.download_as_text()
                reader = csv.reader(io.StringIO(existing_file))
                existing_data = list(reader)
            except Exception as e:
                logger.error(f"Error reading existing file: {e}")
                existing_data = [header]
        else:
            existing_data = [header]
            logger.info(f"Creating new file {FILE_NAME} in bucket {BUCKET_NAME}")

        # Prevent duplicates
        if data not in existing_data:
            existing_data.append(data)

        # Write data back to GCS
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(existing_data)

        blob.upload_from_string(output.getvalue(), content_type="text/csv")
        logger.info(f"Feedback appended to {BUCKET_NAME}/{FILE_NAME}")
        return True
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}", exc_info=True)
        return False

class PredictionInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predictions: str

class FeedbackInput(BaseModel):
    review: str
    predicted_sentiment: str
    actual_sentiment: str
    rating: str

# Routes
@app.get("/", response_class=FileResponse)
async def get_homepage():
    return "static/index.html"

@app.post("/submit-feedback")
async def submit_feedback(feedback: FeedbackInput):
    """Handles storing user feedback in a JSON file."""
    try:
        logger.info(f"Received feedback: {feedback}")
        result = store_feedback(
            feedback.review,
            feedback.predicted_sentiment,
            feedback.actual_sentiment,
            feedback.rating
        )
        logger.info(f"Feedback storage result: {result}")
        return JSONResponse(content={"message": "Feedback saved successfully!"}, status_code=200)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in feedback submission: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e), "details": error_details}, status_code=500)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionInput):
    """Handles prediction requests by preprocessing text, tokenizing, padding, and predicting sentiment."""
    input_text = preprocess_text(request.text)

    # Tokenize and pad sequence
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)

    # Make prediction
    predictions = model.predict(padded_sequence)
    predicted_class = np.argmax(predictions, axis=1)[0]
    sentiment = ["Negative", "Neutral", "Positive"][predicted_class]
    
    return PredictionResponse(predictions=sentiment)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
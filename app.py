import os
import string
import pickle
import numpy as np
import uvicorn
import nltk
import logging
import pandas as pd
import io
import csv
import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from google.cloud import storage
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable, ClassificationQualityMetric, ClassificationClassBalance,
    ClassificationConfusionMatrix, ClassificationRocCurve
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('punkt_tab')


# Constants
BUCKET_NAME = "sentiment-reports-bucket"
FILE_NAME = "Current_User_Reviews.csv"
MODEL_DIR = "Models"

# Stopwords set for text preprocessing
lemmatizer = WordNetLemmatizer()
stopwords = {
    'a', 'an', 'the', 'i', 'he', 'she', 'it', 'they', 'we', 'you', 'her', 'him', 'his',
    'their', 'theirs', 'your', 'yours', 'my', 'mine', 'our', 'ours', 'at', 'by', 'for',
    'in', 'on', 'to', 'from', 'with', 'between', 'through', 'while', 'during',
    'under', 'until', 'over', 'both', 'either', 'neither', 'nor', 'or', 'is', 'was',
    'are', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'have', 'has', 'had',
    'shall', 'will', 'can', 'may', 'might', 'must'
}


# Load LSTM Model
def load_model():
    with open(f"{MODEL_DIR}/lstm_model.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(f"{MODEL_DIR}/lstm_model.weights.h5")
    return model

model = load_model()

# Load Tokenizer
def load_tokenizer():
    with open(f"{MODEL_DIR}/tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)

tokenizer = load_tokenizer()

app = FastAPI()


# Preprocessing function
def preprocess_text(text: str) -> str:
    """Preprocesses input text by applying lowercasing, tokenization, punctuation removal,
    stopword removal, numeric filtering, and stemming."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if not word.isdigit()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


# Store feedback in Google Cloud Storage
def store_feedback(review, predicted_sentiment, actual_sentiment, rating):
    """Stores user feedback in a CSV file on GCS."""
    feedback_data = [review, predicted_sentiment, actual_sentiment, rating]
    return upload_csv_to_gcs(feedback_data)


# Upload feedback to GCS
def upload_csv_to_gcs(data):
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(FILE_NAME)
        header = ["review", "prediction", "target", "rating"]

        existing_data = []
        if blob.exists():
            existing_file = blob.download_as_text()
            existing_data = list(csv.reader(io.StringIO(existing_file)))
        else:
            existing_data.append(header)

        print(f"\n****uploading data***:\n{tuple(data)}")
        if tuple(data) not in [tuple(row) for row in existing_data[1:]]:
            existing_data.append(data)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(existing_data)

        blob.upload_from_string(output.getvalue(), content_type="text/csv")
        logger.info("Feedback uploaded successfully")

        check_model_drift()

        return True
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        return False

# Check Model Drift
def check_model_drift():
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        ref_blob = bucket.blob("train.csv")
        current_blob = bucket.blob(FILE_NAME)

        sentiment_to_capital={0:'Negative',1:'Neutral',2:'Positive',
                   'Negative':'Negative','Neutral':'Neutral','Positive':'Positive',
                   'positive':'Positive','negative':'Negative','neutral':'Neutral'
                   }

        if not ref_blob.exists() or not current_blob.exists():
            logger.warning("Required data files missing. Skipping drift check.")
            return

        reference_data = pd.read_csv(io.StringIO(ref_blob.download_as_text()))
        current_data = pd.read_csv(io.StringIO(current_blob.download_as_text()))

        current_data['prediction'] = current_data['prediction'].apply(lambda x: sentiment_to_capital[x])
        current_data['target'] = current_data['target'].apply(lambda x: sentiment_to_capital[x])
            
        logger.info(f'Reference data shape: {reference_data.shape}')
        logger.info(f'Current data shape: {current_data.shape}')
        
        current_data.rename(columns={'review': 'Text'}, inplace=True)
        current_data.drop(columns=['rating'], inplace=True)
        
        print(f"current_data columns:{current_data.columns}\n reference_data columns:{reference_data.columns}")

        report = Report(metrics=[
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
            ClassificationConfusionMatrix(),
            DataDriftTable(num_stattest='kl_div', cat_stattest='psi')
        ])

        report.run(reference_data=reference_data, current_data=current_data)


        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
            report.save_html(temp_file.name)
            temp_file_path = temp_file.name
        
     
        report_blob = bucket.blob("reports/drift_report_latest.html")
        with open(temp_file_path, 'rb') as file_obj:
            report_blob.upload_from_file(file_obj, content_type='text/html')
            
        os.remove(temp_file_path)
        logger.info("Drift check completed successfully")  


    except Exception as e:

        logger.error(f"Error in drift checking: {str(e)}", exc_info=True)


        


@app.get("/view-drift-report", response_class=HTMLResponse)
async def view_drift_report():
    """Serves the drift report HTML directly."""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("reports/drift_report_latest.html")
        
        if not blob.exists():
            return HTMLResponse(content="<html><body><h1>No drift report available yet</h1></body></html>")
        
        # Download the report content
        report_content = blob.download_as_text()
        
        # Serve the HTML directly
        return HTMLResponse(content=report_content)
    except Exception as e:
        logger.error(f"Error serving drift report: {str(e)}", exc_info=True)
        return HTMLResponse(content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>")



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
        print("\nresult1:\n",feedback.review,feedback.predicted_sentiment,feedback.actual_sentiment,feedback.rating)
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
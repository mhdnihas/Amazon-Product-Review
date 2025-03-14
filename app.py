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
from nltk.stem import WordNetLemmatizer
import pandas as pd
import json
from google.cloud import storage
import csv
import io
import logging
from evidently.report import Report
from evidently.metrics import DataDriftTable, ClassificationQualityMetric



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('wordnet')

stop_words=set(stopwords.words('english'))
negation_words = {"not", "no", "never", "wasn't", "isn't", "doesn't", "don't", "didn't", "haven't", "hadn't", "couldn't", 
                  "shouldn't", "won't", "wouldn't", "shan't", "can't", "mustn't","off","some","but","than","too","few","very"}

stop_words.difference_update(negation_words)

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

lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """Preprocesses input text by applying lowercasing, tokenization, punctuation removal,
    stopword removal, numeric filtering, and stemming."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if not word.isdigit()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
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
        check_model_drift()
        return True
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}", exc_info=True)
        return False


def check_model_drift():
    try:
        logger.info('Checking model drift')
        
        # Use Google Cloud Storage client directly
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        # Download reference data
        ref_blob = bucket.blob("train.csv")
        if not ref_blob.exists():
            logger.warning("Reference data file doesn't exist - skipping drift check")
            return
            
        ref_content = ref_blob.download_as_text()
        reference_data = pd.read_csv(io.StringIO(ref_content))
        
        # Download current data
        current_blob = bucket.blob(FILE_NAME)
        if not current_blob.exists():
            logger.warning(f"File {FILE_NAME} doesn't exist - skipping drift check")
            return
            
        current_content = current_blob.download_as_text()
        current_data = pd.read_csv(io.StringIO(current_content))
        
        logger.info(f'Reference data shape: {reference_data.shape}')
        logger.info(f'Current data shape: {current_data.shape}')
        
        # Create a mapping dictionary for sentiment values
        sentiment_to_num = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
            # Handle capitalized versions too
            'Negative': 0,
            'Neutral': 1,
            'Positive': 2
        }
        
        # Map column names if they don't match exactly
        current_data_mapped = current_data.copy()
        if 'predicted_sentiment' in current_data.columns and 'prediction' not in current_data.columns:
            current_data_mapped['prediction'] = current_data['predicted_sentiment'].map(sentiment_to_num)
            
        if 'actual_sentiment' in current_data.columns and 'target' not in current_data.columns:
            current_data_mapped['target'] = current_data['actual_sentiment'].map(sentiment_to_num)
        
        common_cols = ['prediction', 'target']
        
        if all(col in reference_data.columns for col in common_cols) and \
           all(col in current_data_mapped.columns for col in common_cols):
            
            ref_subset = reference_data[common_cols]
            current_subset = current_data_mapped[common_cols]
            
            # Generate report
            report = Report(metrics=[ClassificationQualityMetric()])
            report.run(reference_data=ref_subset, current_data=current_subset)
            
            # First save the report to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
                report.save_html(temp_file.name)
                temp_file_path = temp_file.name

            os.makedirs('static/reports', exist_ok=True)
            local_report_path = 'static/reports/drift_report_latest.html'
            report.save_html(local_report_path)
            logger.info(f'Drift report saved locally at: {local_report_path}')
        
     
            # Then upload it to GCS
            report_blob = bucket.blob("reports/drift_report_latest.html")
            with open(temp_file_path, 'rb') as file_obj:
                report_blob.upload_from_file(file_obj, content_type='text/html')
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            
            # Get a signed URL for the report that will work in browsers
            report_url = report_blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(hours=24),
                method="GET"
            )
            
            logger.info(f'Drift report generated and uploaded to GCS. URL: {report_url}')
            return report_url
            
        else:
            logger.error('Datasets do not have compatible columns for drift analysis')
            logger.info(f'Reference columns: {reference_data.columns.tolist()}')
            logger.info(f'Current columns: {current_data_mapped.columns.tolist()}')
            
    except Exception as e:
        logger.error(f"Error in drift checking: {str(e)}", exc_info=True)


        

@app.get("/drift-report")
async def get_drift_report():
    """Returns the URL to the latest drift report."""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("reports/drift_report_latest.html")
        
        if not blob.exists():
            return JSONResponse(content={"error": "No drift report available yet"}, status_code=404)
        
        report_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=24),
            method="GET"
        )
        
        return JSONResponse(content={"report_url": report_url}, status_code=200)
    except Exception as e:
        logger.error(f"Error generating drift report URL: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.get("/local-drift-report", response_class=FileResponse)
async def get_local_drift_report():
    """Serves the locally saved drift report."""
    report_path = "static/reports/drift_report_latest.html"
    if os.path.exists(report_path):
        return FileResponse(report_path)
    else:
        return JSONResponse(
            content={"error": "No local drift report available"}, 
            status_code=404
        )
    
    

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
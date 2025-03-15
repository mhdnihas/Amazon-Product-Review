from datetime import datetime, timedelta
import threading
import time
import logging
from google.cloud import storage
import io
import pandas as pd


BUCKET_NAME = "amazon-product-review"
FILE_NAME = "Current_User_Reviews.csv"



DRIFT_THRESHOLD = 0.10 
MIN_FEEDBACK_COUNT = 300 
FEEDBACK_VOLUME_THRESHOLD = 600  
MIN_DAYS_BETWEEN_RETRAINS = 7  

last_retrain_date = None
retraining_lock = threading.Lock()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def should_retrain(drift_score, feedback_count):
    """Decision function to determine if retraining is needed"""
    global last_retrain_date
    
    # Don't retrain if we did it recently
    if last_retrain_date and datetime.now() - last_retrain_date < timedelta(days=MIN_DAYS_BETWEEN_RETRAINS):
        logger.info(f"Recent retraining detected ({(datetime.now() - last_retrain_date).days} days ago). Skipping.")
        return False
        
    # Not enough data yet
    if feedback_count < MIN_FEEDBACK_COUNT:
        logger.info(f"Not enough feedback data yet ({feedback_count}/{MIN_FEEDBACK_COUNT}). Skipping retraining.")
        return False
        
    # Case 1: High drift detected
    if abs(drift_score) > DRIFT_THRESHOLD and feedback_count >= MIN_FEEDBACK_COUNT:
        logger.info(f"Significant drift detected ({drift_score}). Triggering retraining.")
        return True
        
    # Case 2: Enough volume of new data
    if feedback_count >= FEEDBACK_VOLUME_THRESHOLD:
        logger.info(f"Volume threshold reached ({feedback_count} feedback entries). Triggering retraining.")
        return True
        
    return False



def trigger_retraining():
    """Handles the retraining process"""
    global last_retrain_date
    
    # Use lock to prevent multiple retraining processes
    if not retraining_lock.acquire(blocking=False):
        logger.info("Retraining already in progress. Skipping.")
        return False
    
    try:
        # 1. Download training data
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        # Get original training data
        train_blob = bucket.blob("train.csv")
        train_content = train_blob.download_as_text()
        train_df = pd.read_csv(io.StringIO(train_content))
        
        # Get feedback data
        feedback_blob = bucket.blob(FILE_NAME)
        feedback_content = feedback_blob.download_as_text()
        feedback_df = pd.read_csv(io.StringIO(feedback_content))
        
        logger.info(f"Downloaded training data: {train_df.shape} and feedback data: {feedback_df.shape}")
        
        # 2. Prepare data for retraining
        # Map text labels to numbers if needed
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2,
                         'Negative': 0, 'Neutral': 1, 'Positive': 2}
        
        feedback_df['target'] = feedback_df['target'].map(sentiment_map)
        
        # 3. Combine datasets (use only validated feedback where prediction != target)
        misclassified_feedback = feedback_df[feedback_df['prediction'] != feedback_df['target']]
        logger.info(f"Found {len(misclassified_feedback)} misclassified samples for focused retraining")
        
        # 4. Retrain the model (simplified example)
        # In a real implementation, you'd load your model architecture and retrain
        # with the combined dataset, potentially using transfer learning
        
        # For this example, we'll just log that we would retrain
        logger.info("Model would be retrained here with combined dataset")
        
        # 5. Evaluate the new model against a validation set
        
        # 6. If the new model performs better, save it
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        # model.save_weights(f"Models/lstm_model.weights_{model_version}.h5")
        
        # 7. Update the current model
        # with open("Models/lstm_model.json", "w") as json_file:
        #     json_file.write(model.to_json())
        # model.save_weights("Models/lstm_model.weights.h5")
        
        # 8. Update last retrain date
        last_retrain_date = datetime.now()
        
        # 9. Save a backup of the feedback data and reset if needed
        backup_blob = bucket.blob(f"feedback_backups/feedback_{model_version}.csv")
        backup_blob.upload_from_string(feedback_content, content_type="text/csv")
        
        logger.info(f"Retraining completed successfully. New model version: {model_version}")
        return True
        
    except Exception as e:
        logger.error(f"Error during retraining: {e}", exc_info=True)
        return False
    finally:
        retraining_lock.release()

print("last_retrain_date:",last_retrain_date)
print("should retrian:",should_retrain(0.1, 400))  # False
last_retrain_date = datetime.now()
print("last_retrain_date:",last_retrain_date)
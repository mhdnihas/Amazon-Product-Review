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
from evidently.report import Report
from evidently.metrics import DataDriftPresetMetric


nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')


app = FastAPI()



with open("Models/lstm_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)


model.load_weights("Models/lstm_model.weights.h5")


with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


FEEDBACK_FILE = "feedback.json"



def log_predictions(review, predicted_sentiment):
    log_data = {
        "review": review,
        "predicted_sentiment": predicted_sentiment
    }
    with open("predictions.json", "a") as f:
        f.write(json.dumps(log_data) + "\n")


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



def store_feedback(review, predicted_sentiment, actual_sentiment=None, rating=None):
    """Stores user feedback in a JSON file."""
    feedback_data = {
        "review": review,
        "predicted_sentiment": predicted_sentiment,
        "actual_sentiment": actual_sentiment,
        "rating": rating
    }

    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(feedback_data)

    with open(FEEDBACK_FILE, "w") as file:
        json.dump(data, file, indent=4)





class PredictionInput(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    predictions: str


@app.get("/", response_class=FileResponse)
async def get_homepage():
    return "static/index.html"



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


    # Store feedback if actual sentiment and rating are provided
    if request.actual_sentiment and request.rating:
        store_feedback(request.text, sentiment, request.actual_sentiment, request.rating)

          
    return PredictionResponse(predictions=sentiment)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


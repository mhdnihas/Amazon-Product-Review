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



# Download necessary NLTK data files
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')


# Initialize FastAPI app
app = FastAPI()



# Load the model architecture from JSON file
with open("Models/lstm_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)


# Load the model weights
model.load_weights("Models/lstm_model.weights.h5")

# Load the tokenizer
with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize NLP tools
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
          
    return PredictionResponse(predictions=sentiment)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


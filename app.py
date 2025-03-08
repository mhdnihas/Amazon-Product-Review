from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = FastAPI()

# Load the model architecture from JSON file
with open("Models/lstm_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

# Print model summary
print(model.summary())

# Load the model weights
model.load_weights("Models/lstm_model.weights.h5")

# Load the tokenizer
with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class PredictionInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predictions: str

@app.get("/", response_class=FileResponse)
async def get_homepage():
    return "static/index.html"

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionInput):
    input_text = request.text
    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([input_text])
    # Pad the sequence
    max_len = 200  # Same as used during training
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    # Predict
    predictions = model.predict(padded_sequence)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    if predicted_class == 0:
        sentiment = "Negative"
    elif predicted_class == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    
    return PredictionResponse(predictions=sentiment)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
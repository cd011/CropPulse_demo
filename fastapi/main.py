import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

#load environment variables
load_dotenv()

app = FastAPI()

MODEL = tf.keras.models.load_model(os.getenv("MODEL_PATH"))
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return "Hello, world"

def read_as_numpy(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = read_as_numpy(await file.read())
    
    predicted_label = MODEL.predict(np.expand_dims(img, 0))
    predicted_class = CLASS_NAMES[np.argmax(predicted_label[0])]
    confidence = round(100*(np.max(predicted_label[0])),2)
    return predicted_class, confidence
    

if __name__ == "__main__":
    HOST=os.getenv("HOST", "0.0.0.0")
    PORT=int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=HOST, port=PORT)
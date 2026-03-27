from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Dummy Soil Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128,128,3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

soil_types = ["Sandy", "Clay", "Loamy", "Black", "Red"]

@app.get("/")
def home():
    return {"message": "Smart Agri AI Running 🚀"}


@app.post("/predict-soil")
async def predict_soil(file: UploadFile = File(...)):
    image = Image.open(file.file).resize((128,128))
    img = np.array(image)/255.0
    img = img.reshape(1,128,128,3)

    pred = model.predict(img)
    soil = soil_types[np.argmax(pred)]

    return {"soil_type": soil}


@app.post("/recommend-crop")
def recommend_crop():
    return {
        "recommended_crops": [
            "Rice 🌾",
            "Wheat 🌱",
            "Maize 🌽"
        ]
    }

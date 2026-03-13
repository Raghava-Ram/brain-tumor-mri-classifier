from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- Workaround for Keras 3 to Keras 2 Dense layer compatibility ---
_original_dense_init = tf.keras.layers.Dense.__init__
def _custom_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = _custom_dense_init
# -------------------------------------------------------------------

app = FastAPI(title="Brain Tumor Classification API", debug=True)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and classes
model = None
class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model('Pretrained_model.keras')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((380, 380)) # Resize to target size used in training
    if image.mode != "RGB":
        image = image.convert("RGB") # Ensure 3 channels
    image_array = np.array(image) # Convert PIL image to numpy array
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
    return image_array

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Standard prediction endpoint accepting an uploaded image.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)

        if model is None:
            raise RuntimeError("Model is not loaded")

        predictions = model.predict(processed_image)
        predicted_class_index = int(np.argmax(predictions))
        predicted_class = class_labels[predicted_class_index]
        confidence = float(np.max(predictions) * 100)
        raw_probs = {
            class_labels[i].capitalize(): float(prob * 100)
            for i, prob in enumerate(predictions[0])
        }

        return JSONResponse(
            content={
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": raw_probs,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("brain_tumor_CNN.h5")
class_labels = ["Tumor", "No Tumor"]  # or ["Yes", "No"] based on your dataset

# Preprocessing function
def predict_mri(image):
    image = image.resize((128, 128))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return f"{predicted_class} ({confidence:.2%} confidence)"

# Gradio UI
gr.Interface(
    fn=predict_mri,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="🧠 Brain Tumor Detection"
).launch()

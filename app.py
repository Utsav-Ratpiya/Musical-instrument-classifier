import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("cnn_model.h5")

# IMPORTANT:
# Keep class order EXACTLY same as training dataset order
class_names = [
    "acordian",
    "alphorn",
    "bagpipes",
    "banjo",
    "bongo_drum",
    "casaba",
    "castanets",
    "clarinet",
    "clavichord",
    "concertina",
    "Didgeridoo",
    "drums",
    "dulcimer",
    "flute",
    "guiro",
    "guitar",
    "harmonica",
    "harp",
    "marakas",
    "ocarina",
    "piano",
    "saxaphone",
    "sitar",
    "steel drum",
    "Tambourine",
    "trombone",
    "trumpet",
    "tuba",
    "violin",
    "Xylophone"
]

# Prediction function
def predict(img):

    # Convert image to RGB
    img = img.convert("RGB")

    # Resize image to training size
    img = img.resize((128, 128))

    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize image
    img_array = img_array.astype("float32") / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)

    # Get predicted index
    predicted_index = np.argmax(prediction)

    # Get confidence score
    confidence = float(np.max(prediction))

    # Get predicted class name
    predicted_class = class_names[predicted_index]

    # Confidence threshold
    if confidence < 0.50:
        return "⚠️ Model is not confident. Try a clearer image."

    # Return result
    return f"""
🎵 Predicted Instrument: {predicted_class}
📊 Confidence Score: {confidence:.2f}
"""

# Create Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🎵 Musical Instrument Classifier",
    description="Upload an image of a musical instrument to classify it using a CNN Deep Learning model.",
    theme="soft"
)

# Launch application
interface.launch()
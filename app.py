import gradio as gr
from transformers import pipeline

# Load a pre-trained deepfake detection model
deepfake_model = pipeline("image-classification", model="umm-maybe/dfdc")

def analyze_image(img):
    predictions = deepfake_model(img)
    return {pred['label']: float(pred['score']) for pred in predictions}

# Gradio Interface
iface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="filepath"),
    outputs="label",
    title="Deepfake Detection Tool",
    description="Upload an image to check for deepfake or manipulation."
)

if __name__ == "__main__":
    iface.launch()

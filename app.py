import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# --- Fix for Keras 3 compatibility ---
_original_dense_init = tf.keras.layers.Dense.__init__
def _custom_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = _custom_dense_init
# ------------------------------------

# Download model
model_path = hf_hub_download(
    repo_id="Raghava-Ram/brain-tumor-efficientnet",
    filename="pretrained_model.keras"
)

model = tf.keras.models.load_model(model_path)

class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']


def predict(image):
    if image is None:
        return None, None, None

    image = image.resize((380, 380))

    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    pred_class = class_labels[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)

    probs = {
        class_labels[i]: float(preds[0][i])
        for i in range(len(class_labels))
    }

    return pred_class, confidence, probs


# Custom CSS for better styling
css = """
.main-header {
    text-align: center;
    color: var(--body-text-color);
    margin-bottom: 20px;
}
.sub-header {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin-bottom: 40px;
}
.output-panel {
    background-color: var(--bg-color);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid var(--border-color-primary);
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif']
)

with gr.Blocks(theme=theme, css=css) as interface:
    gr.Markdown(
        """
        <div class="main-header">
            <h1>🧠 Brain Tumor Classification</h1>
        </div>
        <div class="sub-header">
            <h3>Upload an MRI scan to detect and classify brain tumors using deep learning.</h3>
            <p>Our model can identify Glioma, Meningioma, Pituitary tumors, or confirm the absence of a tumor.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📸 Upload Scan")
                image_input = gr.Image(type="pil", label="MRI Image", height=380)
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Analyze Image", variant="primary")
                    
        with gr.Column(scale=1):
            with gr.Group(elem_classes="output-panel"):
                gr.Markdown("### 📊 Analysis Results")
                with gr.Row():
                    pred_output = gr.Text(label="Predicted Class")
                    conf_output = gr.Number(label="Confidence (%)")
                
                gr.Markdown("#### Class Probabilities")
                prob_output = gr.Label(label="Distribution", num_top_classes=4)
                
    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[pred_output, conf_output, prob_output]
    )
    
    def clear_outputs():
        return None, None, None, None
        
    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[image_input, pred_output, conf_output, prob_output]
    )
    
    gr.Markdown(
        """
        <br><hr>
        <p style="text-align: center; color: var(--body-text-color-subdued); font-size: 0.9em;">
            <strong>Disclaimer:</strong> This tool is for educational and research purposes only. 
            It is not intended to be a substitute for professional medical advice, diagnosis, or treatment.
        </p>
        """
    )

interface.launch()
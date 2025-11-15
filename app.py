import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize

from models.hybrid_model import HybridModel
from explainability.gradcam import GradCAM

st.set_page_config(page_title="🧠 Alzheimer MRI Classifier", layout="centered")
st.title("🧠 Alzheimer’s Disease Classification (MRI + Grad-CAM)")

labels = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    device = "cpu"
    model = HybridModel(num_classes=4, pretrained=True).to(device)

    # Make sure the file name matches your GitHub file
    model.load_state_dict(torch.load("models/model_epoch_1.pth", map_location=device))

    model.eval()
    return model, device


model, device = load_model()


# -----------------------------
# File Upload Section
# -----------------------------
uploaded = st.file_uploader("📤 Upload MRI slice (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Load as grayscale
    img = Image.open(uploaded).convert("L")
    img_arr = np.array(img)

    # Resize to match training data (128x128)
    img_resized = resize(img_arr, (128, 128))

    # Convert to torch tensor
    x = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # -----------------------------
    # Correct Grad-CAM Target Layer
    # -----------------------------
    target_layer = model.cnn.layer4[-1].conv3   # correct last conv of ResNet50

    gradcam = GradCAM(model, target_layer)

    # Prediction
    out = model(x)
    probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
    idx = np.argmax(probs)
    pred_label = labels[idx]

    # Generate CAM
    cam, _ = gradcam.generate(x)

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = cv2.cvtColor(np.uint8(img_resized * 255), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    # Display Images
    st.image(
        [img, overlay],
        caption=["Original MRI", f"Grad-CAM Highlight → {pred_label}"],
        width=300
    )

    # Prediction Label
    st.success(f"🧠 Predicted Stage: **{pred_label}**")

    # Confidence Scores
    st.subheader("📊 Prediction Confidence")
    st.bar_chart({labels[i]: float(probs[i]) for i in range(4)})

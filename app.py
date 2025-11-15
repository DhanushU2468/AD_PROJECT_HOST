import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize

from models.hybrid_model import HybridModel
from explainability.gradcam import GradCAM

st.set_page_config(page_title="🧠 Alzheimer’s MRI Classifier", layout="centered")
st.title("🧠 Alzheimer’s Disease Classification (MRI + Grad-CAM)")

labels = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridModel(num_classes=4, pretrained=False).to(device)

    # Load model from GitHub folder: models/
    model.load_state_dict(torch.load("models/model_epoch_1.pth", map_location=device))

    model.eval()
    return model, device

model, device = load_model()

# ----------------------------
# Upload Image
# ----------------------------
uploaded = st.file_uploader("📤 Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    img_arr = np.array(img)

    # Resize MRI to match training resolution
    img_resized = resize(img_arr, (128, 128))

    # Model tensor
    x = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # ----------------------------
    # Grad-CAM: pick last conv layer
    # ----------------------------
    target_layer = model.cnn.layer4[-1].conv3
    gradcam = GradCAM(model, target_layer)

    # Generate CAM
    cam, idx = gradcam.generate(x)
    pred_label = labels[idx]

    # ----------------------------
    # Create Heatmap Overlay
    # ----------------------------
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    orig = cv2.cvtColor(np.uint8(img_resized * 255), cv2.COLOR_GRAY2BGR)

    # Resize heatmap to match original MRI
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    # Overlay
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    # ----------------------------
    # Display Result
    # ----------------------------
    st.image([img, overlay],
             caption=["🧠 Original MRI", f"🔥 Heatmap (Prediction: {pred_label})"],
             width=300)

    st.success(f"### ✅ Predicted: **{pred_label}**")

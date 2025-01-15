import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

from trashcan_core.components.data_loader import CustomDataLoader
from trashcan_core.components.constants import (
    MATERIAL_VERSION_PATH,
    N_CLASSES,
    HUES,
    INPUT_DIMS,
)

BATCH_SIZE = 1
onnx_model_path = "C:\\Users\\mikol\\PythonProjects\\TrashCan\\models\\resunet34v8.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

data_loader = CustomDataLoader(MATERIAL_VERSION_PATH, batch_size=BATCH_SIZE)
data_loader.crop(*INPUT_DIMS)


def overlay_hues_on_image(original_image, predicted_probs):
    hsv_image = np.zeros((*INPUT_DIMS, 3), dtype=np.uint8)

    predicted_classes = np.argmax(predicted_probs[0], axis=0)

    for class_idx in range(N_CLASSES - 1):
        hue = HUES.get(f"class_{class_idx}", 0)
        mask = predicted_classes == class_idx
        hsv_image[..., 0][mask] = hue
        hsv_image[..., 1][mask] = 255
        hsv_image[..., 2][mask] = 255

    rgb_mask = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)

    overlay_image = cv2.addWeighted(original_image, 0.7, rgb_mask, 0.3, 0)
    return overlay_image


st.title("Trashcan Segmentation")

if "current_image_index" not in st.session_state:
    st.session_state.current_image_index = 0


def load_image(index):
    image_tensor, _ = data_loader.dataset[index]
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1)
    return image_tensor, image_np


current_index = st.session_state.current_image_index
image_tensor, image_np = load_image(current_index)

st.subheader(f"Original Image (Index: {current_index})")
st.image(image_np, caption="Original Image", use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Previous Image"):
        st.session_state.current_image_index -= 1
        if st.session_state.current_image_index < 0:
            st.session_state.current_image_index = len(data_loader.dataset) - 1
        st.rerun()
with col2:
    if st.button("Next Image"):
        st.session_state.current_image_index += 1
        if st.session_state.current_image_index >= len(data_loader.dataset):
            st.session_state.current_image_index = 0
        st.rerun()

if st.button("Generate Segmentation"):
    image_numpy = image_tensor.numpy().astype(np.float32)
    image_numpy = np.expand_dims(image_numpy, axis=0)

    outputs = ort_session.run(None, {"input": image_numpy})

    predicted_probs = outputs[0]

    overlay_image = overlay_hues_on_image(
        (image_np * 255).astype(np.uint8), predicted_probs
    )
    st.subheader("Segmented Overlay")
    st.image(
        overlay_image, caption="Overlay with Segmentation", use_container_width=True
    )

    overlay_pil = Image.fromarray(overlay_image)
    st.download_button(
        label="Download Segmented Image",
        data=overlay_pil.tobytes(),
        file_name=f"segmented_image_{current_index}.png",
        mime="image/png",
    )

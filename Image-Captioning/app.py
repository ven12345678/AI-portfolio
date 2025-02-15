import streamlit as st
from PIL import Image
from caption import generate_caption

st.title("🖼️ AI Image Captioning App ^^")
st.write("Upload an image, and AI will generate a caption for it!")

uploaded_file = st.file_uploader("Choose an imgae...", type = ["jpg", "png", "jpeg"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption = "Uploaded Image")

  st.write("⏳ Generating caption...")
  caption = generated_caption(uploaded_file)
  st.success(f"**Caption:** {caption}")

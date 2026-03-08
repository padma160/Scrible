import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import enhance_image, extract_text
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key="AIzaSyBUKL_BRDWZTjiZRdb-WDs4CxVkkwY815k")

st.title("📝 Scribble to Digital")
st.write("Convert messy handwritten notes into clean text & to-do lists")

uploaded_file = st.file_uploader("Upload notes image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    enhanced = enhance_image(img_array)

    st.image(enhanced, caption="Enhanced Image", use_column_width=True)

    with st.spinner("🔍 Extracting text with OCR..."):
        raw_text = extract_text(enhanced)

    st.subheader("📄 Raw OCR Text")
    st.text(raw_text)

    if st.button("✨ Convert to Digital"):
        with st.spinner("🤖 Processing with AI..."):
            prompt = f"""
            Clean this OCR text, correct spelling using context,
            and extract to-do tasks separately.

            OCR Text:
            {raw_text}

            Output format:
            Clean Notes:
            - ...

            To-Do List:
            - ...
            """

            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            result = response.text

            st.subheader("✅ Digital Output")
        st.text(result.replace('\n', ' ').strip())

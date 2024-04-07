import io
import json

import httpx
import pandas as pd
import streamlit as st

st.title("Image Text Retrieval Playground")

st.divider()
st.header("Search for an image with a description below")

st.text_input("Write an image description", key="image_description")


if image_description := st.session_state.image_description:
    st.write(f"Searching for images similar to the description: {image_description}")
    response = httpx.post(url="http://0.0.0.0:8000/retrieve_image_from_text", json={"text": image_description})
    st.image(io.BytesIO(response.content))

st.divider()
st.header("Search for captions with an image")

st.file_uploader("Upload an image to search for captions", key="uploaded_image")
uploaded_image = st.session_state.uploaded_image

if uploaded_image:
    st.image(uploaded_image)
    response = httpx.post(url="http://0.0.0.0:8000/retrieve_text_from_image", files={"file": uploaded_image})
    predictions = json.loads(response.content)["predictions"]
    st.write(pd.DataFrame(predictions))

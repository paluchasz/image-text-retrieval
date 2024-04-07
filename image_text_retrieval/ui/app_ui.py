import io

import httpx
import streamlit as st

st.text_input("Write an image description", key="image_description")


if image_description := st.session_state.image_description:
    st.write(f"Searching for images similar to the description: {image_description}")
    result = httpx.post(url="http://0.0.0.0:8000/retrieve_image_from_text", json={"text": image_description})
    st.image(io.BytesIO(result.content))

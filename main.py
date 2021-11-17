import streamlit as st
from PIL import Image
import numpy as np

st.title("automatic license plate recognition".upper())
st.sidebar.title("SETTINGS")
st.sidebar.markdown("-" * 20)
st.sidebar.markdown("Upload license plate image  here!".upper())
file_uploader = st.sidebar.file_uploader("", ["jpg", "png", "jpeg", "gif"])

if file_uploader is not None:
    image = file_uploader.read()
    st.image(image, use_column_width=True)


import streamlit as st
from PIL import Image

from predict import predict

st.title("Dog or Wolf")

upload_file = st.file_uploader("Please upload your image here!", type=["jpg", "png"])

if upload_file is not None:
    image = Image.open(upload_file)

    st.image(image, caption="Uploaded Image", width=256)
    result = predict(image)

    if result == 0:
        st.write("Dog!")
    elif result == 1:
        st.write("Wolf!")
    else:
        st.write("Something went wrong. Please try again.")

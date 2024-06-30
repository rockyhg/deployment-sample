import requests
import streamlit as st
from PIL import Image

st.title("Dog or Wolf")

upload_file = st.file_uploader("Please upload your image here!", type=["jpg", "png"])

if upload_file is not None:
    image = Image.open(upload_file)

    st.image(image, caption="Uploaded Image", width=256)

    # FASTAPIサーバーに画像を送信
    files = {"file": upload_file.getvalue()}
    response = requests.post("http://localhost:8000/predict", files=files)  # ローカル用
    # response = requests.post("https://xxx.onrender.com/predict", files=files)  # 本番用

    # 応答の内容をコンソールに表示
    print("Response status code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error reading JSON response:", e)

    # 応答として受け取った予測結果を表示
    if response.status_code == 200:
        result = response.json()
        if result["result"] == 0:
            st.write("Dog!")
        elif result["result"] == 1:
            st.write("Wolf!")
        else:
            st.write("Something went wrong. Please try again.")
    else:
        st.write("Something went wrong. Please try again.")

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("📷 人脸检测系统（作业版）")
st.subheader("基于 OpenCV 实现，无需 face_recognition")

# 上传图片
uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 画框
    for (x, y, w, h) in faces:
        cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.success(f"🔍 检测到 {len(faces)} 张人脸！")
    st.image(Image.fromarray(img_np), caption="检测结果")
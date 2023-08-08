import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(os.path.join(parent_path, 'utils'))
from face_verification_utilities import *

st.set_page_config(page_title='E-KYC')
col1, col2 = st.columns(2)
uploaded_face_image = st.file_uploader("Choose an image of your face")

uploaded_face_identity_card_image = st.file_uploader("Choose an image of your face on indentity card")

if uploaded_face_image is not None and uploaded_face_identity_card_image is not None:
    extension_face_image = uploaded_face_image.name.split(".")[-1] in ("jpg", "jpeg", "png")
    extension_face_identity_image = uploaded_face_identity_card_image.name.split(".")[-1] in ("jpg", "jpeg", "png")
    if (not extension_face_image) and (not extension_face_identity_image):
        raise ValueError("Image must be jpg or png format!")
    # Convert the file to an opencv image.
    face_img_bytes = np.asarray(bytearray(uploaded_face_image.read()), dtype=np.uint8)
    opencv_image_face = cv2.imdecode(face_img_bytes, 1)
    face_identity_card_img_bytes = np.asarray(bytearray(uploaded_face_identity_card_image.read()), dtype=np.uint8)
    opencv_image_face_identity_card = cv2.imdecode(face_identity_card_img_bytes, 1)

if st.button('Verify'):
    score, cropped_enhanced_face_1, cropped_enhanced_face_2 = get_similarity_score(opencv_image_face, opencv_image_face_identity_card)
    if score < 0.5:
        st.write(score)
        st.write("We detect that 2 faces are highly of the same person")
    with col1:
        st.header("Before and after processing")
        st.image([cv2.cvtColor(opencv_image_face, cv2.COLOR_BGR2RGB), cropped_enhanced_face_1], width=250)
    with col2:
        st.header("Before and after processing")
        st.image([cv2.cvtColor(opencv_image_face_identity_card, cv2.COLOR_BGR2RGB), cropped_enhanced_face_2], width=250)
    

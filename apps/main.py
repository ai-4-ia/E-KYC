import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
from fastapi import FastAPI, File, UploadFile
import uvicorn
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(os.path.join(parent_path, 'utils'))
from face_verification_utilities import *

###This is part testing using streamlit###
# st.set_page_config(page_title='E-KYC')
# uploaded_face_image = st.file_uploader("Choose an image of your lived face")

# uploaded_face_identity_card_image = st.file_uploader("Choose an image of your face on indentity card")

# if uploaded_face_image is not None and uploaded_face_identity_card_image is not None:
#     extension_face_image = uploaded_face_image.name.split(".")[-1] in ("jpg", "jpeg", "png")
#     extension_face_identity_image = uploaded_face_identity_card_image.name.split(".")[-1] in ("jpg", "jpeg", "png")
#     if (not extension_face_image) and (not extension_face_identity_image):
#         raise ValueError("Image must be jpg or png format!")
#     # Convert the file to an opencv image.
#     face_img_bytes = np.asarray(bytearray(uploaded_face_image.read()), dtype=np.uint8)
#     opencv_image_face = cv2.imdecode(face_img_bytes, 1)
#     face_identity_card_img_bytes = np.asarray(bytearray(uploaded_face_identity_card_image.read()), dtype=np.uint8)
#     opencv_image_face_identity_card = cv2.imdecode(face_identity_card_img_bytes, 1)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.header("Your uploaded face image")
#         st.image(cv2.cvtColor(opencv_image_face, cv2.COLOR_BGR2RGB), width=250)
#     with col2:
#         st.header("Your uploaded identity card")
#         st.image(cv2.cvtColor(opencv_image_face_identity_card, cv2.COLOR_BGR2RGB), width=250)

# if st.button('Verify'):
#     st.write("Phase 2: Liveness face validate with passive technique!!!")
#     face_area_depth_map, liveness_val = predict_liveness(opencv_image_face)
#     if liveness_val == "Lived":
#         st.write("Based on depth map area around your face")
#         st.write("We detect that this is your lived face!!!")
#         st.image(face_area_depth_map, clamp=True, width=250)
#         st.write("")
#         st.write("Phase 3: Deepfake detection!!!")
#         lived_face_deepfake_score = deepfake_prediction(opencv_image_face)
#         identity_face_deepfake_score = deepfake_prediction(opencv_image_face_identity_card)
#         if lived_face_deepfake_score < 0.2 and identity_face_deepfake_score < 0.2:
#             st.write("Based on the scoring, we detect both the images have not been tampered by deepfake!!!")
#             st.write("Your face score: ", lived_face_deepfake_score)
#             st.write("Your identity face score: ", identity_face_deepfake_score)
#             st.write("Phase 4: Verify identity based on facial features!!!")
#             face_verification = FacialAttributeVerification()
#             score, cropped_enhanced_face_1, cropped_enhanced_face_2 = face_verification.get_similarity_score(opencv_image_face, opencv_image_face_identity_card)
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.header("After processing your lived face image")
#                 st.image( cropped_enhanced_face_1, width=250)
#             with col2:
#                 st.header("After processing your identity card face")
#                 st.image(cropped_enhanced_face_2, width=250)
#             if score < 0.5:
#                 st.write(score)
#                 st.write("We detect that 2 faces are highly of the same person!!!")
#             st.write("Phase 5: Identity card reading!!!")
#             identity_card_information = identity_card_reading(opencv_image_face_identity_card)
#             st.write(identity_card_information)
#         else:
#             st.write("Based on the scoring, we detect either images have been tampered by deepfake!!!")
#             st.write("Your face score: ", lived_face_deepfake_score)
#             st.write("Your identity face score: ", identity_face_deepfake_score)
#     else:
#         st.write("Based on depth map area around your face")
#         st.write("We detect that this is not your lived face!!!")
#         st.image(face_area_depth_map, clamp=True, width=250)

### This is part deploy as API using FastAPI
app = FastAPI()

@app.post("/face_verification")
async def face_verification(live_face_img: UploadFile = File(...), cccd_img: UploadFile = File(...)):
    # Convert byte image to opencv format
    live_face_img_content = await live_face_img.read()
    face_img_bytes = np.asarray(bytearray(live_face_img_content), dtype=np.uint8)
    opencv_image_face = cv2.imdecode(face_img_bytes, 1)
    
    cccd_img_content = await cccd_img.read()
    cccd_img_bytes = np.asarray(bytearray(cccd_img_content), dtype=np.uint8)
    opencv_cccd_img = cv2.imdecode(cccd_img_bytes, 1)
    
    # Verify identity
    _, liveness_val = predict_liveness(opencv_image_face)
    if liveness_val == "Lived":
        lived_face_deepfake_score = deepfake_prediction(opencv_image_face)
        identity_face_deepfake_score = deepfake_prediction(opencv_cccd_img)
        if lived_face_deepfake_score < 0.2 and identity_face_deepfake_score < 0.2:
            face_verification = FacialAttributeVerification()
            score, _, _ = face_verification.get_similarity_score(opencv_image_face, opencv_cccd_img)
            return score
        else:
            return "The uploaded images properly tampered by deepfake"
    else:
        return "Your face image is not lived image"

@app.post("/identity_card_reader")
async def identity_card_reader(cccd_img: UploadFile = File(...)):
    # Convert byte image to opencv format
    cccd_img_content = await cccd_img.read()
    cccd_img_bytes = np.asarray(bytearray(cccd_img_content), dtype=np.uint8)
    opencv_cccd_img = cv2.imdecode(cccd_img_bytes, 1)
    
    # Verify identity
    identity_card_information = identity_card_reading(opencv_cccd_img)
    return identity_card_information
    
    
if __name__ == "__main__":
    # uvicorn.run(app, debug=True)
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False, debug=False)
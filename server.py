import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt

from model import preprocess, crop, model
from model2 import preprocess2, model2, crop_image_from_gray

st.set_page_config(
    page_title="MBK Vision - DR Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Diabetic Retinopathy Stages Prediction")


col1, col2, col3 = st.columns([3,2,2], gap="large")


with col1:

    uploaded_file = st.file_uploader("Upload a retinal photo", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        col_select, col_button = st.columns([3, 1], gap="large")

        with col_select:
            selected_model = st.selectbox('Select a Model', ("Resnet50-1", "Resnet50-2"))

        with col_button:
            st.write("")
            st.write("")
            button = st.button("Predict")
            

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
    
        if button:
            if selected_model == "Resnet50-1":
                preprocessed_img = preprocess(opencv_image)
                prediction = model.predict(np.expand_dims(preprocessed_img,axis=0))[0]
            elif selected_model == "Resnet50-2":
                preprocessed_img = preprocess2(opencv_image)
                prediction = model2.predict(np.expand_dims(preprocessed_img,axis=0))[0]

            no_dr = float(format(prediction[0]*100,".2f"))
            mild = float(format(prediction[1]*100,".2f"))
            moderate = float(format(prediction[2]*100,".2f"))
            severe = float(format(prediction[3]*100,".2f"))
            pdr = float(format(prediction[4]*100,".2f"))

            chart_data = pd.DataFrame({
                "Stages":["None","Mild","Moderate", "Severe", "Proliferative"],
                "predictions":[no_dr,mild,moderate, severe, pdr]
            })

            chart = alt.Chart(chart_data).mark_bar(width=50).encode(
                x=alt.X("Stages", sort=["None","Mild","Moderate", "Severe", "Proliferative"],
                axis=alt.Axis(labelAngle=45, labelAlign='left')),
                y=alt.Y("predictions", title="Predictions (%)"),
                color=alt.Color("predictions", scale=alt.Scale(
                    domain=[0, 20, 90, 100],
                    range=['green', 'blue', 'blue', 'red']
                ))
            ).properties(
                title="Prediction Results",
 
            ).configure_axis(
                labelFontSize=14,
                titleFontSize=16
            ).configure_title(
                fontSize=20
            )
            st.altair_chart(chart, use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.markdown('<div style="text-align:center">Original Image</div>', unsafe_allow_html=True)
        preprocessed_img = crop_image_from_gray(opencv_image)
        st.image(preprocessed_img, channels="BGR")

with col3:
    if uploaded_file is not None:
        if button:
            st.markdown('<div style="text-align:center">Preprocessed Image</div>', unsafe_allow_html=True)
            if selected_model == "Resnet50-1":
                preprocessed_img = preprocess(opencv_image)       
                st.image(preprocessed_img, channels="RGB")
            elif selected_model == "Resnet50-2":
                preprocessed_img = preprocess2(opencv_image)       
                st.image(preprocessed_img, channels="RGB")

        



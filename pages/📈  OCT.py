import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt


from oct.model import model, preprocess, crop

st.set_page_config(
    page_title="MBK Vision - OCT Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.header("OCT (Optical Coherence Tomography) Prediction")

col1, col2 = st.columns([3,4], gap="large")


prediction_result = ""
percentage = 0


with col1:
    uploaded_file = st.file_uploader("Upload a oct photo", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        
        button = st.button("Predict")
            
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
    
        if button:
            preprocessed_img = preprocess(opencv_image)
            prediction = model.predict(np.expand_dims(preprocessed_img,axis=0))[0]
    
            CNV = float(format(prediction[0]*100,".2f"))
            DME = float(format(prediction[1]*100,".2f"))
            DRUSEN = float(format(prediction[2]*100,".2f"))
            NORMAL = float(format(prediction[3]*100,".2f"))

            predictions_tuple = {"CNV":CNV, "DME": DME, "DRUSEN":DRUSEN, "NORMAL":NORMAL}
            max_key, max_value = max(predictions_tuple.items(), key=lambda x: x[1])

            if max_key == "CNV":
                prediction_result = "Choroidal Neovascularization (CNV)"
                percentage = max_value
            elif max_key == "DME":
                prediction_result = "Diabetic Macular Edema (DME)"
                percentage = max_value 
            elif max_key == "DRUSEN":
                prediction_result = "DRUSEN"
                percentage = max_value
            elif max_key == "NORMAL":
                prediction_result = "NORMAL"
                percentage = max_value
            else:
                st.write("An error occured!")
            


            chart_data = pd.DataFrame({
                "Stages":["CNV","DME","DRUSEN", "NORMAL"],
                "predictions":[CNV,DME,DRUSEN, NORMAL]
            })

            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("Stages", sort=["NORMAL","CNV","DME", "DRUSEN"],
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
        preprocessed_img = crop(opencv_image)
        st.image(preprocessed_img, channels="BGR")
        if button:
            st.metric(label=prediction_result, value=f"{percentage} %")
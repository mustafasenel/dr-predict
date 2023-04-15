import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt


st.set_page_config(
    page_title="MBK Vision - DR Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Diabetic Retinopathy Stages Prediction")

with st.sidebar:
    st.title("MBK Vision - DEMO")
    st.subheader("Diabetic Retinopathy and")
    st.subheader("Optical Coherence Tomography Prediction")



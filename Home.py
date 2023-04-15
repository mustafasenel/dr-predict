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

st.title("DEMO APPLICATION")

st.header("📊 Diabetic Retinopathy Stages Prediction ")
st.header("📈 OCT (Optical Coherence Tomography) Prediction")
with st.sidebar:
    st.title("MBK Vision - DEMO")




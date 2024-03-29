import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import altair as alt


st.set_page_config(
    page_title="MBK Vision - DR Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.header("📊 Diabetic Retinopathy Stages Prediction ")
st.subheader("What is DR (Diabetic Retinopathy)")
st.write("Diabetic retinopathy is a disease that affects the retina at the back of the eye due to damage caused by high blood sugar levels. The blood vessels in the retina can become damaged, leading to vision loss. Early symptoms of the disease include blurry vision, faded colors, and dark spots in the visual field. Diabetic retinopathy is the most common eye disease in people with diabetes, occurring in about half of them. Early diagnosis of the disease is vital to prevent vision loss and provide more treatment options.")

st.subheader("Machine Learning and DR")
st.write("Machine learning can be used as an effective tool in diagnosing this disease. By analyzing eye images, machine learning algorithms can identify the symptoms of the disease and be used for diagnosis. This method can be an alternative for fast and accurate diagnosis, and early diagnosis of diabetic retinopathy can provide better treatment and management.")

st.subheader("🗄️ Datasets")

aptos_image = Image.open("./assets/aptos.png")

#DATASET SECTION


dr1 = Image.open("./assets/dr1.jpg")
dr2 = Image.open("./assets/dr2.jpg")
dr3 = Image.open("./assets/dr3.jpg")
dr4 = Image.open("./assets/dr4.jpg")



st.image(aptos_image)
st.write("""One of the most widely used machine learning datasets for diabetic retinopathy diagnosis is the APTOS dataset. This dataset consists of 3,435 retinal images obtained from different imaging devices. The images are labeled with five different severity grades of diabetic retinopathy, ranging from 0 (no DR) to 4 (proliferative DR).

Other commonly used datasets for diabetic retinopathy diagnosis include the Kaggle Diabetic Retinopathy Detection dataset, the Messidor-2 dataset, the IDRiD dataset and the EyePACS dataset. Each of these datasets has its own unique characteristics and challenges, and researchers may choose to use different datasets depending on their specific needs and research goals.

In conclusion, machine learning models trained on diabetic retinopathy datasets can play a crucial role in early detection and management of this serious eye disease. The APTOS dataset, along with other commonly used datasets, provides a valuable resource for researchers and clinicians working on this important public health issue.
"""
)

st.subheader("🎚️ Diabetic Retinopathy Stages")

data_col1, data_col2, data_col3, data_col4 = st.columns([1,1,1,1])

with data_col1:
    st.markdown('<div style="text-align:center">Mild</div>', unsafe_allow_html=True)
    st.image(dr1)
with data_col2:
    st.markdown('<div style="text-align:center">Moderate</div>', unsafe_allow_html=True)
    st.image(dr2)
with data_col3:
    st.markdown('<div style="text-align:center">Severe</div>', unsafe_allow_html=True)
    st.image(dr3)
with data_col4:    
    st.markdown('<div style="text-align:center">Proliferative</div>', unsafe_allow_html=True)
    st.image(dr4)

st.subheader("📈 Data Preprocessing")
st.write("""Image data used for diagnosing Diabetic Retinopathy (DR) undergoes pre-processing steps such as improving image quality, removing noise, performing image segmentation, and feature extraction.

The reason for the need for pre-processing is that direct image processing algorithms do not have high success rates in DR diagnosis. Images are sometimes captured under poor lighting conditions, from different angles, and using various imaging devices. Therefore, image pre-processing steps can help improve the performance of machine learning models used for DR diagnosis.

Methods used in image pre-processing may include:

Image brightness adjustment
Noise reduction
Contrast enhancement
Image segmentation
Image alignment
All of these operations are performed to improve image quality and enhance the performance of machine learning models used for DR diagnosis.
"""
)

original_image = Image.open("./assets/original-dr.jpg")
preprocessed_image = Image.open("./assets/pre-dr.jpg")
preprocessed_image_2 = Image.open("./assets/pre-dr-2.jpg")

image_col_1, image_col_2, image_col_3 = st.columns([1,1,1])

with image_col_1:
    st.subheader("ORIGINAL")
    st.image(original_image)
with image_col_2:
    st.subheader("MODEL-1 PREPROCESSED")
    st.image(preprocessed_image_2)

with image_col_3:
    st.subheader("MODEL-2 PREPROCESSED")
    st.image(preprocessed_image)

st.subheader("👨‍💻 Model Training")
st.write("""After the preprocessing steps, the next step in diagnosing diabetic retinopathy (DR) is to train a machine learning model. In my project, I used the ResNet50 architecture to train the model. The images were resized to 400x400 pixels after preprocessing, and the Keras library was used for training. The training was performed with a batch size of 2, 1000 steps per epoch, and 15 epochs.

The ResNet50 architecture is a deep convolutional neural network that has achieved state-of-the-art performance in many computer vision tasks. It consists of 50 layers and uses residual connections to address the problem of vanishing gradients during training.

Resizing the images to a smaller size helped to reduce the computational requirements during training without significantly affecting the performance of the model. The Keras library provided an easy-to-use interface for building and training the model, allowing us to focus on the development of the preprocessing steps and the selection of the appropriate architecture.

Training the model for 15 epochs with a batch size of 2 and 1000 steps per epoch resulted in a model with good performance in detecting diabetic retinopathy. However, it is important to note that the performance of the model can be further improved with the use of more advanced techniques, such as data augmentation and transfer learning.

In conclusion, the use of ResNet50 architecture, image resizing, and the Keras library resulted in an effective machine learning model for diagnosing diabetic retinopathy. By continuing to refine and improve the model, we can make significant progress in the early detection and treatment of this disease.
"""
)

st.title("Model Comparison")

# History pickle dosyasını yükle
with open('./dr/history-1.pkl', 'rb') as file:
    history = pickle.load(file)

#MODEL-2 HISTORY
with open('./dr/history-2.pkl', 'rb') as file:
    history_2 = pickle.load(file)
# Extract accuracy and validation accuracy from history
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

# en son epoch'ta accuracy ve loss değerlerini al
last_epoch = len(history['accuracy'])

# DataFrame oluştur
data = pd.DataFrame({'last_accuracy': acc[last_epoch-1],
                     'last_val_accuracy': val_acc[last_epoch-1],
                     'last_loss': loss[last_epoch-1],
                     'last_val_loss': val_loss[last_epoch-1]}, index=['model-1'])


###################################### MODEL - 2#############################
# Extract accuracy and validation accuracy from history
acc_2 = history_2['accuracy']
val_acc_2 = history_2['val_accuracy']
loss_2 = history_2['loss']
val_loss_2 = history_2['val_loss']

# en son epoch'ta accuracy ve loss değerlerini al
last_epoch = len(history_2['accuracy'])

# DataFrame oluştur
data_2 = pd.DataFrame({'last_accuracy': acc_2[last_epoch-1],
                     'last_val_accuracy': val_acc_2[last_epoch-1],
                     'last_loss': loss_2[last_epoch-1],
                     'last_val_loss': val_loss_2[last_epoch-1]}, index=['model-2'])

# İki DataFrame'i birleştirme
result = pd.concat([data, data_2])

# CSS to inject contained in a string
st.dataframe(result, use_container_width=True)

df = pd.DataFrame({
    'Epoch': range(1, len(acc)+1),
    'Accuracy': acc,
    'Validation Accuracy': val_acc,
    'Loss': loss,
    'Validation Loss': val_loss
})

# MODEL -1 HISTORY

st.title("Model-1 History")

chart_col_1, chart_col_2 = st.columns([1,1], gap="medium")

with chart_col_1:
    # Add a title to the chart using st.header()
    st.header("📈 Accuracy Chart")

    # Plot the history using st.line_chart()
    st.line_chart(df.set_index('Epoch')[['Accuracy', 'Validation Accuracy']])
with chart_col_2:
    st.header("📈 Loss Chart")

    # Plot the history using st.line_chart()
    st.line_chart(df.set_index('Epoch')[['Loss', 'Validation Loss']])


#MODEL-2 HISTORY
st.title("Model-2 History")

df_2 = pd.DataFrame({
    'Epoch': range(1, len(acc_2)+1),
    'Accuracy': acc_2,
    'Validation Accuracy': val_acc_2,
    'Loss': loss_2,
    'Validation Loss': val_loss_2
})

chart_col_1, chart_col_2 = st.columns([1,1], gap="medium")

with chart_col_1:


    # Add a title to the chart using st.header()
    st.header("📈 Accuracy Chart")

    # Plot the history using st.line_chart()
    st.line_chart(df_2.set_index('Epoch')[['Accuracy', 'Validation Accuracy']])

with chart_col_2:
    # Add a title to the chart using st.header()
    st.header("📈 Loss Chart")

    # Plot the history using st.line_chart()
    st.line_chart(df_2.set_index('Epoch')[['Loss', 'Validation Loss']])










st.header("📈 OCT (Optical Coherence Tomography) Prediction")
with st.sidebar:
    st.title("MBK Vision - DEMO")




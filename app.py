import streamlit as st
import numpy as np, tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
st.title('MNIST Handwritten Digit Recognition')
st.write('Upload a 28x28 grayscale image or draw one and predict the digit.')
uploaded = st.file_uploader('Upload PNG/JPG of digit (28x28 recommended)', type=['png','jpg','jpeg'])
model = None
if st.button('Load model'):
    model = tf.keras.models.load_model('models/mnist_cnn_tf')
    st.success('Model loaded.')
if uploaded is not None:
    img = Image.open(uploaded).convert('L').resize((28,28))
    st.image(img, caption='input (resized to 28x28)', width=150)
    arr = np.array(img)
    if model is None:
        st.warning('Load the model first (click Load model) or run train.py to generate model.')
    else:
        pred, conf = -1, -1.0
        try:
            from src.predict import predict_image
            pred, conf = predict_image(model, arr)
        except Exception as e:
            st.error(str(e))
        st.write(f'Predicted digit: **{pred}** (conf: {conf:.3f})')

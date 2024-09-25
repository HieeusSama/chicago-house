import streamlit as st
import numpy as np
import pickle

import os


# Load các mô hình đã huấn luyện
with open('linear_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('ridge_model.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('mlp_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)

# Giao diện người dùng với Streamlit
st.title('Dự đoán giá nhà')

# Lựa chọn mô hình
model_option = st.selectbox(
    'Chọn mô hình dự đoán',
    ('Linear Regression', 'Ridge Regression', 'Neural Network')
)

# Nhận các input từ người dùng
bedroom = st.number_input('Số phòng ngủ', min_value=0, max_value=10, value=3)
space = st.number_input('Diện tích (m²)', min_value=0.0, max_value=1000.0, value=100.0)
room = st.number_input('Số phòng', min_value=0, max_value=10, value=5)
lot = st.number_input('Diện tích lô đất (m²)', min_value=0.0, max_value=10000.0, value=300.0)
tax = st.number_input('Thuế (VNĐ)', min_value=0.0, max_value=10000000.0, value=1000000.0)
bathroom = st.number_input('Số phòng tắm', min_value=0, max_value=5, value=2)
garage = st.number_input('Số chỗ để xe', min_value=0, max_value=5, value=1)
condition = st.slider('Tình trạng nhà (1-10)', min_value=1, max_value=10, value=5)

# Chuyển các input thành mảng numpy
input_features = np.array([[bedroom, space, room, lot, tax, bathroom, garage, condition]])

# Nút bấm dự đoán
if st.button('Dự đoán giá nhà'):
    if model_option == 'Linear Regression':
        prediction = linear_model.predict(input_features)
    elif model_option == 'Ridge Regression':
        prediction = ridge_model.predict(input_features)
    else:
        prediction = mlp_model.predict(input_features)

    st.write(f'Giá nhà dự đoán: {round(prediction[0], 2)} VNĐ')

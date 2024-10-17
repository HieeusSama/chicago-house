import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

model_dir = os.path.join(os.path.dirname(__file__), 'models')
lr_model = joblib.load(os.path.join(model_dir, 'linear_regression_model.joblib'))
ridge_model = joblib.load(os.path.join(model_dir, 'ridge_regression_model.joblib'))
mlp_model = joblib.load(os.path.join(model_dir, 'mlp_regressor_model.joblib'))
stacking_model = joblib.load(os.path.join(model_dir, 'stacking_regressor_model.joblib'))

models = {
    'Linear Regression': lr_model,
    'Ridge Regression': ridge_model,
    'Neural Network': mlp_model,
    'Stacking Regressor': stacking_model
}

st.title("Ứng Dụng Dự Đoán Giá Nhà")

st.header("Nhập thông tin để dự đoán")

bedroom = st.number_input('Số phòng ngủ', value=0)
space = st.number_input('Diện tích (m2)', value=0)
room = st.number_input('Số phòng', value=0)
lot = st.number_input('Diện tích lô đất (m2)', value=0)
tax = st.number_input('Thuế ($)', value=0)
bathroom = st.number_input('Số phòng tắm', value=0)
garage = st.number_input('Chỗ để xe (số xe)', value=0)
condition = st.selectbox('Tình trạng', [0, 1])

model_name = st.selectbox(
    'Chọn mô hình',
    ['Linear Regression', 'Ridge Regression', 'Neural Network', 'Stacking Regressor']
)

if st.button('Dự đoán'):
    input_data = {
        'Bedroom': bedroom,
        'Space': space,
        'Room': room,
        'Lot': lot,
        'Tax': tax,
        'Bathroom': bathroom,
        'Garage': garage,
        'Condition': condition
    }

    input_df = pd.DataFrame([input_data])

    try:
        model = models[model_name]
        prediction = model.predict(input_df)[0]

        y_test = pd.read_csv('y_test.csv')
        X_test = pd.read_csv('X_test.csv')

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.success(f"Giá nhà dự đoán: ${prediction:,.2f}")
        st.info(f"Đánh giá mô hình:\n- MAE: {mae:.4f}\n- R²: {r2:.4f}\n- RMSE: {rmse:.4f}")

    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán: {e}")

# app/ml.py

import tensorflow as tf
import joblib
from streamlit import cache_resource
from constants import STRUCTURES

# Кэшируем модель и скейлеры с учётом структуры
@cache_resource
def load_model_and_scalers(structure):
    model_path = f"./{structure}/model_updated_test_st.keras"
    scalers_path = f"./{structure}/scalers.pkl"
    model = tf.keras.models.load_model(model_path)
    x_scaler, y_scaler = joblib.load(scalers_path)
    return model, x_scaler, y_scaler

def predict_em_properties(model, x_scaler, y_scaler, H, K, L, P):
    import numpy as np
    inp = np.array([[H, K, L, P]], dtype=np.float32)
    scaled = x_scaler.transform(inp)
    pred_scaled = model.predict(scaled)
    se, ghz, refl, absor = y_scaler.inverse_transform(pred_scaled)[0]
    return se, ghz, refl, absor

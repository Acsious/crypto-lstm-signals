import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import logging
import os

SEQ_LENGTH = 120
FORECAST_HORIZON = 30
MODEL_PATH = 'lstm_price_multioutput.h5'
SCALER_PATH = 'scaler_price_multioutput.save'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_lstm_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        logging.error(f"Файл модели не найден: {model_path}")
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
    model = load_model(model_path)
    logging.info(f"Модель успешно загружена из {model_path}")
    return model

def load_scaler(scaler_path: str = SCALER_PATH):
    if not os.path.exists(scaler_path):
        logging.error(f"Файл scaler не найден: {scaler_path}")
        raise FileNotFoundError(f"Scaler не найден по пути: {scaler_path}")
    scaler = joblib.load(scaler_path)
    logging.info(f"Scaler успешно загружен из {scaler_path}")
    return scaler

def prepare_sequence(df: pd.DataFrame, sequence_length: int = SEQ_LENGTH):
    if 'price' not in df.columns:
        raise ValueError('В DataFrame нет колонки price!')
    data = df['price'].values.reshape(-1, 1)
    scaler = load_scaler()
    scaled_data = scaler.transform(data)
    if len(scaled_data) < sequence_length:
        raise ValueError(f"Недостаточно данных для последовательности длиной {sequence_length}")
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    return last_sequence, scaler

def generate_multioutput_predictions(df: pd.DataFrame, model=None, horizon: int = FORECAST_HORIZON) -> np.ndarray:
    if model is None:
        model = load_lstm_model()
    last_sequence, scaler = prepare_sequence(df)
    pred_scaled = model.predict(last_sequence, verbose=0)
    pred_scaled = pred_scaled.reshape(-1, 1)
    pred_real = scaler.inverse_transform(pred_scaled)[:, 0]
    logging.info(f"Сгенерировано {horizon} предсказаний цен (multi-output)")
    return pred_real

if __name__ == "__main__":
    from data_module import load_and_preprocess_data
    df = load_and_preprocess_data('data.csv')
    try:
        preds = generate_multioutput_predictions(df)
        print(f"Предсказанные цены на следующие {FORECAST_HORIZON} периода:")
        print(preds)
    except Exception as e:
        print(f"Ошибка: {e}")

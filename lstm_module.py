import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import talib  
import logging
import os

SEQ_LENGTH = 120
BASE_COLUMNS = ['price', 'market_cap', 'total_volume']
MODEL_PATH = 'bitcoin_lstm_model.h5'
FORECAST_HORIZON = 24

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['price'].values
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['rsi_14'] = talib.RSI(close, timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)
    df['stoch_k'], df['stoch_d'] = talib.STOCH(
        df['price'].values, df['price'].values, df['price'].values,
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    
    df = df.dropna().reset_index(drop=True)
    logging.info(f"Добавлены технические индикаторы. Новая форма DataFrame: {df.shape}")
    return df

def load_lstm_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        logging.error(f"Файл модели не найден: {model_path}")
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
    
    model = load_model(model_path)
    logging.info(f"Модель успешно загружена из {model_path}")
    return model

def prepare_sequences(df: pd.DataFrame, sequence_length: int = SEQ_LENGTH):
    df_with_ind = add_technical_indicators(df.copy())
    feature_columns = BASE_COLUMNS + ['sma_20', 'ema_12', 'rsi_14', 'macd', 'macd_signal', 'macd_hist','bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d']
    data = df_with_ind[feature_columns].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    if len(scaled_data) < sequence_length:
        raise ValueError(f"Недостаточно данных после добавления индикаторов для последовательности длиной {sequence_length}")
    
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, len(feature_columns)))
    return last_sequence, scaler, df_with_ind

def generate_predictions(df: pd.DataFrame, model=None, horizon: int = FORECAST_HORIZON) -> np.ndarray:
    if model is None:
        model = load_lstm_model()
    
    last_sequence, scaler, _ = prepare_sequences(df)
    n_features = last_sequence.shape[2] 
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(horizon):
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0, 0])
        new_row = np.zeros((1, 1, n_features))
        new_row[0, 0, 0] = pred[0, 0]  
        new_row[0, 0, 1:] = current_sequence[0, -1, 1:]
        current_sequence = np.append(current_sequence[:, 1:, :], new_row, axis=1)
    
    predictions_full = np.array(predictions).reshape(-1, 1)
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, 0] = predictions_full[:, 0]
    
    predictions_real = scaler.inverse_transform(dummy)[:, 0]
    
    logging.info(f"Сгенерировано {horizon} предсказаний цен")
    return predictions_real

if __name__ == "__main__":
    from data_module import load_and_preprocess_data
    df = load_and_preprocess_data('data.csv')
    try:
        preds = generate_predictions(df)
        print("Предсказанные цены на следующие 24 периода:")
        print(preds)
    except Exception as e:
        print(f"Ошибка: {e}")
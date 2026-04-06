import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

SEQ_LENGTH = 120
FORECAST_HORIZON = 30
MODEL_PATH = 'lstm_price_multioutput.h5'
SCALER_PATH = 'scaler_price_multioutput.save'

def load_price_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'price' not in df.columns:
        raise ValueError('В файле нет колонки price!')
    return df['price'].values.reshape(-1, 1)

def create_sequences(data, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+horizon].flatten())
    return np.array(X), np.array(y)

def train_lstm_multioutput(csv_path='data.csv',
                          seq_length=SEQ_LENGTH,
                          horizon=FORECAST_HORIZON,
                          model_path=MODEL_PATH,
                          scaler_path=SCALER_PATH):
    data = load_price_data(csv_path)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, scaler_path)
    X, y = create_sequences(data_scaled, seq_length, horizon)
    n = len(X)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    model = Sequential([                            #TODO: вернуть 2 слоя LSTM и Dropout между ними, а также optuna для подбора гиперпараметров
        LSTM(64, input_shape=(seq_length, 1)),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=32, callbacks=callbacks)
    val_loss = model.evaluate(X_val, y_val)
    test_loss = model.evaluate(X_test, y_test)
    print(f'Validation loss: {val_loss:.6f}, Test loss: {test_loss:.6f}')
    print(f'Модель и scaler сохранены: {model_path}, {scaler_path}')

if __name__ == '__main__':
    train_lstm_multioutput()

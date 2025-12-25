import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import talib
import optuna
import logging
import sys

seqLength = 120
START_DATE = '2013-12-28' 
FORECAST_START_DATE = '2025-01-01'  
FORECAST_END_DATE = '2025-02-01'  
MODEL_PATH = 'bitcoin_lstm_model.h5'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_validate_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        required_columns = ['snapped_at', 'price', 'market_cap', 'total_volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in CSV")
        
        df['snapped_at'] = pd.to_datetime(df['snapped_at'])
        
        if df.isnull().any().any():
            logging.warning("Found missing values. Dropping rows with NaN.")
            df = df.dropna()
        
        if (df['price'] <= 0).any():
            logging.warning("Found invalid price values. Dropping rows with non-positive prices.")
            df = df[df['price'] > 0]
        
        df = df.sort_values('snapped_at')
        
        start_date = pd.to_datetime(START_DATE).tz_localize('UTC')
        df = df[df['snapped_at'] >= start_date]
        
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        sys.exit(1)

def add_technical_indicators(df):
    close = df['price'].values
    volume = df['total_volume'].values
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['rsi_14'] = talib.RSI(close, timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)
    df['stoch_k'], df['stoch_d'] = talib.STOCH(close, close, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df = df.dropna()
    return df

def prepare_data(df, sequence_length=seqLength):
    features = ['price', 'market_cap', 'total_volume', 'sma_20', 'ema_12', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d']
    #features = ['price', 'market_cap', 'total_volume'] #версия без индикаторов
    data = df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    total_samples = len(X)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.2)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}) size mismatch")
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError(f"X_val ({X_val.shape[0]}) and y_val ({y_val.shape[0]}) size mismatch")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"X_test ({X_test.shape[0]}) and y_test ({y_test.shape[0]}) size mismatch")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, data[:, 0], df

def build_lstm_model(sequence_length, n_features, units, dropout_rate, learning_rate):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units // 3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def objective(trial, X_train, y_train, X_val, y_val, sequence_length, n_features):
    units = trial.suggest_int('units', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    model = build_lstm_model(sequence_length, n_features, units, dropout_rate, learning_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    val_loss = min(history.history['val_loss'])
    return val_loss

def optimize_hyperparameters(X_train, y_train, X_val, y_val, sequence_length, n_features):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, sequence_length, n_features), 
                   n_trials=20)
    
    best_params = study.best_params
    logging.info(f"Best hyperparameters: {best_params}")
    return best_params

def plot_results(train_data, val_data, test_data, test_predictions, dates, history):
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 1, 1)
    train_end = len(train_data)
    val_end = train_end + len(val_data)
    test_end = val_end + len(test_data)
    
    plt.plot(dates[:train_end], train_data, label='Training Data', color='blue')
    plt.plot(dates[train_end:val_end], val_data, label='Validation Data', color='green')
    plt.plot(dates[val_end:test_end], test_data, label='Test Data', color='red')
    plt.plot(dates[val_end:test_end], test_predictions, label='Test Predictions', color='orange')
    
    plt.title('Bitcoin Price Prediction and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss (MSE)', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='green')
    plt.plot(history.history['mae'], label='Training MAE', color='cyan')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show(block=True)

def main():
    df = load_and_validate_data('data.csv')
    df = add_technical_indicators(df)
    
    sequence_length = seqLength
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, prices, df = prepare_data(df, sequence_length)
    n_features = X_train.shape[2]
    #best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, sequence_length, n_features)
    # model = build_lstm_model(sequence_length, n_features, best_params['units'], best_params['dropout_rate'], best_params['learning_rate'])

    model = build_lstm_model(sequence_length, n_features, 171, 0.1411, 0.000756)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    test_predictions = model.predict(X_test)
    test_predictions_full = np.zeros((test_predictions.shape[0], n_features))
    test_predictions_full[:, 0] = test_predictions[:, 0]
    test_predictions = scaler.inverse_transform(test_predictions_full)[:, 0]
    y_test_full = np.zeros((y_test.shape[0], n_features))
    y_test_full[:, 0] = y_test
    actual_test = scaler.inverse_transform(y_test_full)[:, 0]
    train_data = scaler.inverse_transform(np.hstack((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], n_features-1)))))[:, 0]
    val_data = scaler.inverse_transform(np.hstack((y_val.reshape(-1, 1), np.zeros((y_val.shape[0], n_features-1)))))[:, 0]
    test_data = actual_test
 
    dates = df['snapped_at'].values
    plot_results(train_data, val_data, test_data, test_predictions, dates, history)
    mae = np.mean(np.abs(test_predictions - test_data))
    print(f"\nMean Absolute Error (MAE) on test set: {mae:.2f} USD")
    model.save('bitcoin_lstm_model.h5')

if __name__ == "__main__":
    main()
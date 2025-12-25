import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import talib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['price'].values
    
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['rsi_14'] = talib.RSI(close, timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)
    df['stoch_k'], df['stoch_d'] = talib.STOCH(
        close, close, close,
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    
    df = df.dropna().reset_index(drop=True)
    logging.info(f"Технические индикаторы добавлены. Форма DataFrame: {df.shape}")
    return df

def momentum_strategy(prices):
    rsi_series = talib.RSI(pd.Series(prices), timeperiod=14)
    current_rsi = rsi_series.iloc[-1]
    if current_rsi < 30:
        return 'buy'
    elif current_rsi > 70:
        return 'sell'
    return 'hold'

def trend_following_strategy(prices):
    macd_line, signal_line, _ = talib.MACD(pd.Series(prices))
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        return 'buy'
    elif macd_line.iloc[-1] < signal_line.iloc[-1]:
        return 'sell'
    return 'hold'

def mean_reversion_strategy(prices):
    upper, _, lower = talib.BBANDS(pd.Series(prices), timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    current_price = prices[-1]
    if current_price < lower.iloc[-1]:
        return 'buy'
    elif current_price > upper.iloc[-1]:
        return 'sell'
    return 'hold'

def range_trading_strategy(prices):
    stoch_k, _ = talib.STOCH(prices, prices, prices, fastk_period=14, slowk_period=3, slowd_period=3)
    if stoch_k.iloc[-1] < 20:
        return 'buy'
    elif stoch_k.iloc[-1] > 80:
        return 'sell'
    return 'hold'

def volatility_breakout_strategy(prices):
    series = pd.Series(prices)
    std = series.rolling(20).std()
    upper = series.iloc[-1] + std.iloc[-1] * 1.5
    lower = series.iloc[-1] - std.iloc[-1] * 1.5
    current_price = prices[-1]
    if current_price > upper:
        return 'buy'
    elif current_price < lower:
        return 'sell'
    return 'hold'

# TODO: добавить еще несколько стратегий
strategy_pool_fixed = {
    0: mean_reversion_strategy,          # медвежка
    1: momentum_strategy,                # бычка
    2: mean_reversion_strategy           # боковик
}

def prepare_and_select_classifier(historical_df: pd.DataFrame):
    df_with_ind = add_technical_indicators(historical_df.copy())
    
    returns = df_with_ind['price'].pct_change().shift(-1)
    df_with_ind['Phase'] = np.where(returns > 0.01, 1, np.where(returns < -0.01, 0, 2))
    df_with_ind.dropna(inplace=True)
    
    features = ['rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d']
    
    X = df_with_ind[features]
    y = df_with_ind['Phase']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_classifier = None
    best_accuracy = 0
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        if acc > best_accuracy:
            best_accuracy = acc
            best_classifier = clf
    
    test_acc = accuracy_score(y_test, best_classifier.predict(X_test))
    logging.info(f"Лучший классификатор: {type(best_classifier).__name__}, "
                 f"Val Accuracy: {best_accuracy:.2f}, Test Accuracy: {test_acc:.2f}")
    
    return best_classifier, features, df_with_ind

def classify_phase(classifier, current_data, features):
    # Избегаем предупреждения о feature names, передавая значения без имен
    last_row = current_data[features].iloc[-1:].values
    return classifier.predict(last_row)[0]

def generate_signal(historical_df: pd.DataFrame, lstm_predictions: np.ndarray = None):
    classifier, features, df_with_ind = prepare_and_select_classifier(historical_df.copy())
    
    phase = classify_phase(classifier, df_with_ind, features)
    
    selected_strategy = strategy_pool_fixed[phase]
    strategy_name = selected_strategy.__name__
    
    fixed_params = {
        'momentum_strategy': {'rsi_low': 30, 'rsi_high': 70},
        'mean_reversion_strategy': {'bb_period': 20, 'std_dev': 2.0},
        'trend_following_strategy': {'macd_threshold': 0.0},
        'range_trading_strategy': {'stoch_low': 20, 'stoch_high': 80},
        'volatility_breakout_strategy': {'vol_period': 20, 'vol_multiplier': 1.5}
    }.get(strategy_name, {})
    
    prices = df_with_ind['price'].values
    if lstm_predictions is not None:
        prices = np.append(prices, lstm_predictions)
    
    if len(prices) < 50:  
        logging.warning("Недостаточно данных для расчёта индикаторов. Возвращаем 'hold'.")
        signal = 'hold'
    else:
        if selected_strategy == range_trading_strategy:
            signal = selected_strategy(prices, prices, prices)
        else:
            signal = selected_strategy(prices)
    
    logging.info(f"Фаза: {phase}, Стратегия: {strategy_name}, "
                 f"Фиксированные параметры: {fixed_params}, Сигнал: {signal}")
    
    return signal, fixed_params, phase, strategy_name

if __name__ == "__main__":
    from data_module import load_and_preprocess_data
    df = load_and_preprocess_data('data.csv')
    signal, params, phase, strategy_name = generate_signal(df)
    print(f"Сигнал: {signal}")
    print(f"Фаза рынка: {phase}")
    print(f"Стратегия: {strategy_name}")
    print(f"Параметры: {params}")
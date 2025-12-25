import logging
import pandas as pd
from datetime import datetime
import os

LOG_FILE_CSV = 'signals_log.csv'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_monitoring.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def initialize_signals_log():
    if not os.path.exists(LOG_FILE_CSV):
        columns = [
            'timestamp_utc', 'market_phase', 'phase_description', 'strategy_name',
            'strategy_params', 'signal', 'notes'
        ]
        pd.DataFrame(columns=columns).to_csv(LOG_FILE_CSV, index=False)
        logging.info(f"Создан новый файл лога сигналов: {LOG_FILE_CSV}")

def log_signal(signal: str, phase: int, strategy_name: str, params: dict, notes: str = '') -> None:
    phase_descriptions = {0: 'Bear', 1: 'Bull', 2: 'Sideways'}
    phase_desc = phase_descriptions.get(phase, 'Unknown')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    log_message = (
        f"Сигнал сгенерирован | "
        f"Фаза: {phase_desc} | "
        f"Стратегия: {strategy_name} | "
        f"Параметры: {params} | "
        f"Сигнал: {signal.upper()} | "
        f"Заметки: {notes}"
    )
    logging.info(log_message)
    new_record = {
        'timestamp_utc': timestamp,
        'market_phase': phase,
        'phase_description': phase_desc,
        'strategy_name': strategy_name,
        'strategy_params': str(params), 
        'signal': signal,
        'notes': notes
    }
    try:
        df_new = pd.DataFrame([new_record])
        df_new.to_csv(LOG_FILE_CSV, mode='a', header=False, index=False)
    except Exception as e:
        logging.error(f"Ошибка при записи в CSV-лог сигналов: {str(e)}")

def log_system_event(event_type: str, message: str) -> None:
    full_message = f"Системное событие [{event_type}]: {message}"
    if event_type == 'ERROR':
        logging.error(full_message)
    else:
        logging.info(full_message)

def calculate_simple_performance_metrics(df_historical: pd.DataFrame, last_signal: str) -> dict:
    recent = df_historical['price'].tail(30)
    if len(recent) > 1:
        cumulative_return = (recent.iloc[-1] / recent.iloc[0]) - 1
        return {'last_30d_return': f'{cumulative_return:.2%}'}
    return {}

initialize_signals_log()

if __name__ == "__main__":
    log_system_event('START', 'Тестирование модуля мониторинга')
    test_params = {'rsi_threshold_low': 28, 'rsi_threshold_high': 72}
    log_signal(
        signal='buy',
        phase=1,
        strategy_name='momentum_strategy',
        params=test_params,
        notes='Backtest profit: +5.2%'
    )
    log_system_event('COMPLETED', 'Тест завершен успешно')
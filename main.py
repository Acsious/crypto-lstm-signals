from data_module import load_and_preprocess_data
from lstm_module import generate_predictions
from strategy_module import generate_signal
from execution_module import send_telegram_signal
from monitoring_module import log_signal, log_system_event

def main():
    log_system_event('START', 'Запуск ежедневного цикла торговой системы')
    try:
        df = load_and_preprocess_data('data.csv')
        lstm_predictions = generate_predictions(df)  
        signal, params, phase, strategy_name = generate_signal(df, lstm_predictions)
        log_signal(signal, phase, strategy_name, params, notes=f"Предсказания LSTM: {lstm_predictions[:5]}...")
        send_telegram_signal(signal, phase, strategy_name, params, dry_run=False)
        log_system_event('COMPLETED', 'Цикл завершён успешно')
    except Exception as e:
        log_system_event('ERROR', f'Критическая ошибка в основном цикле: {str(e)}')

if __name__ == "__main__":
    main()
from data_module import load_and_preprocess_data
from lstm_module import generate_multioutput_predictions
from strategy_module import find_trade_points
from execution_module import send_telegram_signal, send_telegram_forecast
from monitoring_module import log_signal, log_system_event
from datetime import datetime, timedelta

def generate_forecast_dates(last_date_str, periods):
    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
    return [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(periods)]

def main(user_type='regular'):
    log_system_event('START', 'Запуск ежедневного цикла торговой системы')
    try:
        df = load_and_preprocess_data('data.csv')
        preds = generate_multioutput_predictions(df)
        last_date_str = str(df['date'].iloc[-1]) if 'date' in df.columns else datetime.now().strftime("%Y-%m-%d")
        forecast_dates = generate_forecast_dates(last_date_str, len(preds))

        buy_points, sell_points, trades = find_trade_points(preds[:5], threshold=0.01)
        send_telegram_signal(buy_points, sell_points, trades, preds[:5], forecast_dates[:5], dry_run=False)

        if user_type == 'premium':
            buy_points_full, sell_points_full, trades_full = find_trade_points(preds, threshold=0.01)
            send_telegram_forecast(buy_points_full, sell_points_full, trades_full, preds, forecast_dates, dry_run=False)

        log_system_event('COMPLETED', 'Цикл завершён успешно')
    except Exception as e:
        log_system_event('ERROR', f'Критическая ошибка в основном цикле: {str(e)}')

if __name__ == "__main__":
    # Для обычного пользователя: main('regular')
    # Для премиум-пользователя: main('premium')
    main('regular')

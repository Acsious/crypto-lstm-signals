import logging
import json
from urllib import request, parse
from datetime import datetime
from strategy_module import find_trade_points

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'          # TODO: Заменить на реальный токен
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID_HERE'              # TODO: Заменить на ID чата/пользователя
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'

def send_telegram_signal(buy_points, sell_points, trades, predictions, dates, dry_run: bool = False) -> bool:
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    signal = 'hold'
    if trades:
        first_trade = trades[0]
        if first_trade['buy_idx'] < first_trade['sell_idx']:
            signal = 'buy'
        else:
            signal = 'sell'
    message = f"""
🚨 Новый торговый сигнал (Bitcoin)
📅 Дата и время: {current_time}
🔥 Рекомендация: **{signal.upper()}**
    """.strip()
    if dry_run:
        logging.info("Режим dry-run: сообщение подготовлено, но не отправлено.")
        logging.info(f"Сообщение:\n{message}")
        return True
    data = parse.urlencode({
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }).encode('utf-8')
    try:
        req = request.Request(TELEGRAM_API_URL, data=data, method='POST')
        req.add_header('Content-Type', 'application/x-www-form-urlencoded')
        with request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            if result.get('ok'):
                logging.info("Сигнал успешно отправлен в Telegram.")
                return True
            else:
                logging.error(f"Ошибка Telegram API: {result.get('description')}")
                return False
    except Exception as e:
        logging.error(f"Исключение при отправке в Telegram: {str(e)}")
        return False


def send_telegram_forecast(buy_points, sell_points, trades, predictions, dates, dry_run: bool = False) -> bool:
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    forecast_str = '\n'.join([f"{dates[i]}: {p:.2f}" for i, p in enumerate(predictions)])
    if trades:
        trades_str = '\n'.join([
            f"Сделка {i+1}: Покупка (дата {dates[t['buy_idx']]}, цена {t['buy_price']:.2f}) -> Продажа (дата {dates[t['sell_idx']]}, цена {t['sell_price']:.2f}), Профит: {t['profit']*100:.2f}%"
            for i, t in enumerate(trades)
        ])
    else:
        trades_str = 'Торговых возможностей не найдено.'
    message = f"""
🌟 Премиум-прогноз (Bitcoin)
📅 Дата и время: {current_time}
📈 Прогноз на период: {forecast_str}
Найдено {len(trades)} сделок (порог 1%): {trades_str}
    """.strip()
    if dry_run:
        logging.info("Режим dry-run: премиум-прогноз подготовлен, но не отправлен.")
        logging.info(f"Сообщение:\n{message}")
        return True
    data = parse.urlencode({
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }).encode('utf-8')
    try:
        req = request.Request(TELEGRAM_API_URL, data=data, method='POST')
        req.add_header('Content-Type', 'application/x-www-form-urlencoded')
        with request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            if result.get('ok'):
                logging.info("Премиум-прогноз успешно отправлен в Telegram.")
                return True
            else:
                logging.error(f"Ошибка Telegram API: {result.get('description')}")
                return False
    except Exception as e:
        logging.error(f"Исключение при отправке премиум-прогноза в Telegram: {str(e)}")
        return False
    
if __name__ == "__main__":
    test_dates = [f"2026-04-{str(i+1).zfill(2)}" for i in range(10)]
    test_predictions = [100000, 101000, 102500, 101500, 103000, 104000, 100500, 150000, 101000, 117000]
    buy_points, sell_points, trades = find_trade_points(test_predictions[:5], threshold=0.01)
    print("\nТестирование send_telegram_signal (dry-run):")
    send_telegram_signal(buy_points, sell_points, trades, test_predictions[:5], test_dates[:5], dry_run=True)

    buy_points_full, sell_points_full, trades_full = find_trade_points(test_predictions, threshold=0.01)
    print("\nТестирование send_telegram_forecast (dry-run):")
    send_telegram_forecast(buy_points_full, sell_points_full, trades_full, test_predictions, test_dates, dry_run=True)

import logging
import json
from urllib import request, parse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'          #TODO: Заменить на реальный токен
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID_HERE'              #TODO: Заменить на ID чата/пользователя
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'

def send_telegram_signal(signal: str, phase: int, strategy_name: str, params: dict, dry_run: bool = False) -> bool:
    phase_descriptions = {0: 'Bear (медвежий)', 1: 'Bull (бычий)', 2: 'Sideways (боковой)'}
    phase_desc = phase_descriptions.get(phase, 'Неизвестная фаза')
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    message = f"""
🚨 Новый торговый сигнал (Bitcoin)

📅 Дата и время: {current_time}
📊 Фаза рынка: {phase_desc}
🧠 Выбранная стратегия: {strategy_name}
⚙️ Параметры стратегии: {json.dumps(params, indent=2, ensure_ascii=False)}

🔥 Рекомендация: **{signal.upper()}**

Система торговли на базе LSTM-модели.
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

if __name__ == "__main__":
    test_signal = 'buy'
    test_phase = 1
    test_strategy = 'momentum_strategy'
    test_params = {'rsi_threshold_low': 28, 'rsi_threshold_high': 72}
    
    print("Тестирование в dry-run режиме:")
    send_telegram_signal(test_signal, test_phase, test_strategy, test_params, dry_run=True)
    
    #### для реальной отправки (после настройки токена и чат_айди)
    # print("\nРеальная отправка:")
    # send_telegram_signal(test_signal, test_phase, test_strategy, test_params, dry_run=False)
import pandas as pd
import logging
import sys

START_DATE = '2013-12-28'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path: str = 'data.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        required_columns = ['snapped_at', 'price', 'market_cap', 'total_volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Отсутствуют обязательные колонки: {required_columns}")
        df['snapped_at'] = pd.to_datetime(df['snapped_at'], utc=True)
        
        if df.isnull().any().any():
            logging.warning("Обнаружены пропущенные значения. Удаление строк с NaN.")
            df = df.dropna()
        
        df = df.sort_values('snapped_at')
        if df['snapped_at'].duplicated().any():
            logging.warning("Обнаружены дубликаты по дате. Удаление дубликатов.")
            df = df.drop_duplicates(subset='snapped_at', keep='last')
        
        start_date = pd.to_datetime(START_DATE).tz_localize('UTC')
        df = df[df['snapped_at'] >= start_date]
        df = df.reset_index(drop=True)
        logging.info(f"Данные успешно загружены и обработаны. Форма DataFrame: {df.shape}")
        logging.info(f"Диапазон дат: с {df['snapped_at'].min()} по {df['snapped_at'].max()}")
        return df
    
    except FileNotFoundError:
        logging.error(f"Файл не найден: {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Ошибка при загрузке или обработке данных: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    df = load_and_preprocess_data('data.csv')
    print(df.head())
    print(df.tail())
    print(df.info())
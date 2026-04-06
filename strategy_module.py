import matplotlib.pyplot as plt
from lstm_module import generate_multioutput_predictions

def find_trade_points(prices, threshold=0.01):
    minima = []  
    maxima = []  
    n = len(prices)
    for i in range(1, n-1):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            minima.append((i, prices[i]))
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            maxima.append((i, prices[i]))
    trades = []
    min_idx = 0
    max_idx = 0
    while min_idx < len(minima) and max_idx < len(maxima):
        if minima[min_idx][0] < maxima[max_idx][0]:
            buy_idx, buy_price = minima[min_idx]
            while max_idx < len(maxima) and maxima[max_idx][0] < buy_idx:
                max_idx += 1
            if max_idx < len(maxima):
                sell_idx, sell_price = maxima[max_idx]
                profit = (sell_price - buy_price) / buy_price
                if profit >= threshold:
                    trades.append({'buy_idx': buy_idx, 'buy_price': buy_price,
                                   'sell_idx': sell_idx, 'sell_price': sell_price,
                                   'profit': profit})
                min_idx += 1
                max_idx += 1
            else:
                break
        else:
            max_idx += 1
    buy_points = [(t['buy_idx'], t['buy_price']) for t in trades]
    sell_points = [(t['sell_idx'], t['sell_price']) for t in trades]
    return buy_points, sell_points, trades

def plot_trades(prices, buy_points, sell_points):
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Predicted Price', color='blue')
    if buy_points:
        buy_idx, buy_price = zip(*buy_points)
        plt.scatter(buy_idx, buy_price, color='green', label='Buy (Entry)', marker='^', s=100)
    if sell_points:
        sell_idx, sell_price = zip(*sell_points)
        plt.scatter(sell_idx, sell_price, color='red', label='Sell (Exit)', marker='v', s=100)
    plt.title('Trade Points on Predicted Prices (MultiOutput)')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from data_module import load_and_preprocess_data
    df = load_and_preprocess_data('data.csv')
    preds = generate_multioutput_predictions(df)
    print("Предсказанные цены:")
    print(preds)

    threshold = 0.0011
    buy_points, sell_points, trades = find_trade_points(preds, threshold=threshold)

    print(f"\nНайдено {len(trades)} сделок (порог {threshold*100:.2f}%):")
    for i, t in enumerate(trades, 1):
        print(f"Сделка {i}: Покупка (индекс {t['buy_idx']}, цена {t['buy_price']:.2f}) -> Продажа (индекс {t['sell_idx']}, цена {t['sell_price']:.2f}), Профит: {t['profit']*100:.2f}%")

    plot_trades(preds, buy_points, sell_points)

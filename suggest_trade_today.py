import pandas as pd
import os
import numpy as np
from utils import get_csv_filename, read_config

def read_stock_data(symbol):
    csv_filename = get_csv_filename(symbol)
    if not os.path.exists(csv_filename):
        print(f"文件 {csv_filename} 不存在")
        return None
    df = pd.read_csv(csv_filename)
    df['time_key'] = pd.to_datetime(df['time_key'])
    return df

def suggest_trades(df):
    # 简单的量化交易模型示例
    latest_data = df.iloc[-1]
    close_price = latest_data['close']
    
    buy_prices = [close_price * (1 - 0.02 * i) for i in range(1, 6)]
    sell_prices = [close_price * (1 + 0.02 * i) for i in range(1, 6)]
    
    return buy_prices, sell_prices

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def moving_average_strategy(df, short_window=40, long_window=100):
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0

    # 创建短期和长期的移动平均线
    signals['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # 计算五日和30日的平均价格
    avg_price_5 = df['close'].rolling(window=5, min_periods=1).mean()
    avg_price_30 = df['close'].rolling(window=30, min_periods=1).mean()

    # 计算支撑线
    support_line = df['low'].rolling(window=30, min_periods=1).min()

    # 评估股票超买还是超卖
    overbought = df['close'] > signals['short_mavg'] * 1.05
    oversold = df['close'] < signals['short_mavg'] * 0.95

    # 根据交易量评估股民情绪
    volume_avg = df['volume'].rolling(window=30, min_periods=1).mean()
    high_volume = df['volume'] > volume_avg * 1.5
    low_volume = df['volume'] < volume_avg * 0.5

    # 计算交易量指数和价格指数
    volume_index = df['volume'] / volume_avg
    price_index = df['close'] / signals['short_mavg']

    # 计算 RSI
    rsi = calculate_rsi(df)

    indicators = {
        'short_mavg': signals['short_mavg'].iloc[-1],
        'long_mavg': signals['long_mavg'].iloc[-1],
        'avg_price_5': avg_price_5.iloc[-1],
        'avg_price_30': avg_price_30.iloc[-1],
        'support_line': support_line.iloc[-1],
        'overbought': overbought.iloc[-1],
        'oversold': oversold.iloc[-1],
        'high_volume': high_volume.iloc[-1],
        'low_volume': low_volume.iloc[-1],
        'volume_index': volume_index.iloc[-1],
        'price_index': price_index.iloc[-1],
        'rsi': rsi.iloc[-1]
    }

    return indicators

def main():
    config = read_config()
    symbols = config['symbols']
    
    for symbol in symbols:
        df = read_stock_data(symbol)
        if df is not None:
            # 使用简单策略
            buy_prices, sell_prices = suggest_trades(df)
            print(f"股票: {symbol}")
            print(f"买入建议: {buy_prices}")
            print(f"卖出建议: {sell_prices}")

            # 使用移动平均线策略
            indicators = moving_average_strategy(df)
            print(f"关键指标:")
            print(f"短期移动平均线: {indicators['short_mavg']:.2f}")
            print(f"长期移动平均线: {indicators['long_mavg']:.2f}")
            print(f"五日平均价格: {indicators['avg_price_5']:.2f}")
            print(f"30日平均价格: {indicators['avg_price_30']:.2f}")
            print(f"支撑线: {indicators['support_line']:.2f}")
            print(f"超买: {'是' if indicators['overbought'] else '否'}")
            print(f"超卖: {'是' if indicators['oversold'] else '否'}")
            print(f"价格指数: {indicators['price_index']:.2f}")
            print(f"高交易量: {'是' if indicators['high_volume'] else '否'}")
            print(f"低交易量: {'是' if indicators['low_volume'] else '否'}")
            print(f"交易量指数: {indicators['volume_index']:.2f}")
            print(f"RSI: {indicators['rsi']:.2f}")

            # 分隔符
            print("=" * 20)

if __name__ == "__main__":
    main()

import pandas as pd
import os
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

def main():
    config = read_config()
    symbols = config['symbols']
    
    for symbol in symbols:
        df = read_stock_data(symbol)
        if df is not None:
            buy_prices, sell_prices = suggest_trades(df)
            print(f"股票: {symbol}")
            print(f"买入建议: {buy_prices}")
            print(f"卖出建议: {sell_prices}")

if __name__ == "__main__":
    main()

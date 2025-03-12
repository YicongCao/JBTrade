from futu import *
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from utils import get_csv_filename, get_last_date_from_csv, read_config  # 修改导入

def fetch_k_line_data(symbol, start_date, end_date):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)  # 根据实际配置修改
    try:
        ret, data, page_req_key = quote_ctx.request_history_kline(
            code=symbol,
            start=start_date,
            end=end_date,
            ktype=KLType.K_DAY,  # 日线
            max_count=1000,  # 足够大的数值确保获取全部数据
        )
        if ret == RET_OK:
            df = pd.DataFrame(data)
            df['time_key'] = pd.to_datetime(df['time_key'])
            return df
        else:
            print("获取数据失败:", data)
            return None
    except Exception as e:
        print("发生错误:", str(e))
        return None
    finally:
        quote_ctx.close()

def save_k_line_data(df, csv_filename):
    if os.path.exists(csv_filename):
        df.to_csv(csv_filename, mode='a', header=False, index=False, columns=[
            'time_key', 'open', 'close', 'high', 'low', 'volume'
        ])
    else:
        df.to_csv(csv_filename, index=False, columns=[
            'time_key', 'open', 'close', 'high', 'low', 'volume'
        ])
    print(f"数据已保存至 {csv_filename}")

def fetch_and_save_k_line_data():
    config = read_config()
    symbols = config['symbols']
    days = config['days']

    for symbol in symbols:
        csv_filename = get_csv_filename(symbol)
        last_date = get_last_date_from_csv(csv_filename)
        
        if last_date:
            start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        df = fetch_k_line_data(symbol, start_date, end_date)
        if df is not None:
            save_k_line_data(df, csv_filename)

if __name__ == "__main__":
    fetch_and_save_k_line_data()
from futu import *
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from utils import get_csv_filename, get_last_date_from_csv, read_config
import akshare as ak


def fetch_k_line_data(symbol, start_date, end_date):
    if symbol.startswith('HK.'):
        # 港股用 futu
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
    elif symbol.startswith('US.'):
        # 美股用 akshare
        code = symbol[3:]
        try:
            # akshare 美股日K
            df = ak.stock_us_daily(symbol=code)
            # akshare 返回的日期格式为 yyyyMMdd，需转为 yyyy-mm-dd
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
            # 统一列名
            df = df.rename(columns={
                'date': 'time_key',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'volume': 'volume',
            })
            # 只保留需要的列
            df = df[['time_key', 'open', 'close', 'high', 'low', 'volume']]
            return df
        except Exception as e:
            print(f"akshare 获取美股数据失败: {symbol}, 错误: {e}")
            return None
    else:
        print(f"不支持的股票代码格式: {symbol}")
        return None

def save_k_line_data(df, csv_filename):
    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename)
        existing_df['time_key'] = pd.to_datetime(existing_df['time_key'])
        
        # 合并数据，去重
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['time_key'], keep='last')
        combined_df.to_csv(csv_filename, index=False, columns=[
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
            last_date_dt = pd.to_datetime(last_date)
            today = datetime.now().date()
            if last_date_dt.date() == today:
                start_date = today.strftime('%Y-%m-%d')
            elif last_date_dt.date() < today:
                start_date = (last_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                raise ValueError("last_date 不能晚于今天")
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        df = fetch_k_line_data(symbol, start_date, end_date)
        if df is not None:
            save_k_line_data(df, csv_filename)

if __name__ == "__main__":
    fetch_and_save_k_line_data()
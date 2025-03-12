import os
import pandas as pd
import json

def get_csv_filename(symbol):
    return f"{symbol.replace('.', '_')}.csv"

def get_last_date_from_csv(csv_filename):
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if not df.empty:
            return pd.to_datetime(df['time_key']).max().strftime('%Y-%m-%d')
    return None

def read_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 校验配置完整性和合法性
    if 'symbols' not in config or not isinstance(config['symbols'], list):
        raise ValueError("配置文件中缺少 'symbols' 或其格式不正确")
    if 'days' not in config or not isinstance(config['days'], int):
        raise ValueError("配置文件中缺少 'days' 或其格式不正确")
    if 'openai_api_key' not in config or not isinstance(config['openai_api_key'], str):
        raise ValueError("配置文件中缺少 'openai_api_key' 或其格式不正确")
    if 'wxwork_webhook_key' not in config or not isinstance(config['wxwork_webhook_key'], str):
        raise ValueError("配置文件中缺少 'wxwork_webhook_key' 或其格式不正确")
    
    return config

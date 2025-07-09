import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils import get_csv_filename, read_config
from update_k_line_hist import fetch_and_save_k_line_data

# ====== 趋势分析参数配置 ======
GROW_CONFIG = {
    'boll_window': 20,                  # 布林线窗口长度（常用20日）
    'boll_std': 2,                      # 布林线上下轨标准差倍数（常用2）
    'uptrend_min_days': 5,              # 上升趋势判定：最近多少天内贴近上轨
    'uptrend_min_touch_ratio': 0.5,     # 上升趋势判定：最近N天内有多少比例天数贴近上轨
    'uptrend_strength_threshold': 0.7,  # 上升趋势判定：趋势强度阈值（越接近1越强）
}

def calc_bollinger_bands(df, window, num_std):
    df = df.copy()
    df['MA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['upper'] = df['MA'] + num_std * df['STD']
    df['lower'] = df['MA'] - num_std * df['STD']
    return df

def detect_bollinger_uptrend(
    df,
    config=GROW_CONFIG
):
    window = config['boll_window']
    num_std = config['boll_std']
    min_days = config['uptrend_min_days']
    min_touch_ratio = config['uptrend_min_touch_ratio']
    strength_threshold = config['uptrend_strength_threshold']
    # 计算布林线
    df = calc_bollinger_bands(df, window, num_std)
    if len(df) < window + min_days:
        return False, 0, None
    recent = df.iloc[-min_days:]
    # 统计最近 min_days 有多少天收盘价接近上轨
    touch = (recent['close'] >= recent['upper'] * 0.98).sum()
    touch_ratio = touch / min_days
    # 趋势强度：收盘价与上轨的平均距离（越小越强）
    strength = 1 - (recent['upper'] - recent['close']).clip(lower=0).mean() / recent['upper'].mean()
    # 只要有一半以上天数贴近上轨，且强度大于阈值
    if touch_ratio >= min_touch_ratio and strength > strength_threshold:
        return True, round(strength, 3), recent
    return False, round(strength, 3), recent

def push_to_wxwork(content, webhook_key):
    import requests
    url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "msgtype": "markdown",
        "markdown": {
            "content": content
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"推送失败: {response.text}")

def calc_support_price(df, window=20):
    # 支撑价可用布林线下轨的最近一日，或近window日最低价均值
    if len(df) < window:
        return None
    # 取最近window日布林线下轨均值
    df_boll = calc_bollinger_bands(df, window, GROW_CONFIG['boll_std'])
    support = df_boll['lower'].iloc[-window:].mean()
    return support

def analyze_grow_trend(stock_data_dict, config, grow_config=GROW_CONFIG):
    results = []
    # 构建 symbol -> name 的映射
    symbol_name_map = {s: n for s, n in zip(config.get('symbols', []), config.get('names', []))} if config.get('names') else {}
    for symbol, df in stock_data_dict.items():
        name = symbol_name_map.get(symbol, symbol)
        if len(df) < 25:
            results.append(f"{name}({symbol}): 数据不足，无法判断趋势")
            continue
        uptrend, strength, recent = detect_bollinger_uptrend(df, config=grow_config)
        # 支撑价
        support = calc_support_price(df, window=grow_config['boll_window'])
        cur_price = df['close'].iloc[-1]
        support_str = f"{cur_price:.2f} / {support:.2f}" if support else "-"
        if uptrend:
            results.append(f"**{name}({symbol})**检测到上升趋势，强度: <font color='red'>{strength}</font>，当前价/支撑价: {support_str}")
        else:
            results.append(f"{name}({symbol}): 没有产生上升趋势，强度: {strength}，当前价/支撑价: {support_str}")
    return results

if __name__ == "__main__":
    # 更新股价数据
    # fetch_and_save_k_line_data()
    # 读取配置和数据
    global_config = read_config()
    symbols = global_config['symbols']
    stock_data_dict = {}
    for symbol in symbols:
        csv_file = get_csv_filename(symbol)
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['time_key'] = pd.to_datetime(df['time_key'])
            stock_data_dict[symbol] = df
        else:
            print(f"未找到 {csv_file}，跳过该股票")
    # 分析趋势并推送
    results = analyze_grow_trend(stock_data_dict, global_config, GROW_CONFIG)
    wxwork_key = global_config.get('wxwork_webhook_key')
    msg = '**布林线趋势分析**\n' + '\n'.join(results)
    print(msg)
    if wxwork_key:
        push_to_wxwork(msg, wxwork_key)

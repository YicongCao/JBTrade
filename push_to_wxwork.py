import pandas as pd
import os
import requests
from utils import get_csv_filename, read_config
from get_stock_analysis import read_stock_data, print_stock_analysis
from ask_deepseek import get_deepseek_response
from datetime import datetime

def push_to_wxwork(content, webhook_key):
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

def main():
    config = read_config()
    symbols = config['symbols']
    api_key = config['openai_api_key']
    webhook_key = config['wxwork_webhook_key']
    days = config['days']
    
    for i, symbol in enumerate(symbols):
        df = read_stock_data(symbol)
        if df is not None:
            analysis_output = print_stock_analysis(symbol, df, n=days)
            analysis_output_short = print_stock_analysis(symbol, df, n=10)
            print(f"分析结果:\n{analysis_output}")
            reasoning, answer = get_deepseek_response(analysis_output, api_key)
            print(f"\nDeepSeek 思考过程:\n{reasoning}")
            print(f"\nDeepSeek 回复:\n{answer}")
            
            # 收集输出内容
            analysis_content = f"## 股票代码: {symbol}\n\n**指标分析:**\n{analysis_output_short}"
            # deepseek_content = f"## 股票代码: {symbol}\n\n**交易建议:**\n{answer}"
            deepseek_content = f"{answer}"
            
            # 推送到企业微信
            push_to_wxwork(analysis_content, webhook_key)
            push_to_wxwork(deepseek_content, webhook_key)
            
            if i < len(symbols) - 1:
                print("=" * 20)

if __name__ == "__main__":
    main()

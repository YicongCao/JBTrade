import pandas as pd
import os
from utils import get_csv_filename, read_config
from get_stock_analysis import read_stock_data, print_stock_analysis
from openai import OpenAI
from datetime import datetime

def get_deepseek_response(content, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.lkeap.cloud.tencent.com/v1",
    )
    stream = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "system", "content": "你是一个经验丰富的恒生科技和纳斯达克投资者，请根据用户提供的股票分析内容，结合金融知识分析走势，了解该股票代码对应公司的历史背景、发展阶段和行业特点，分析并给出不同仓位下包含买入卖出价格的每日交易建议，并且在最终结论中，以长期看好科技股票、并期望保持一定持仓的同时逐渐做低持仓成本的倾向，直接计算出我今天该以多少价格、多少比例的仓位、执行买入还是卖出该股票。"},
            {"role": "user", "content": content}
        ],
        stream=True
    )

    reasoning_content = ""
    answer_content = ""
    is_answering = False
    chunk_count = 0  # 记录chunk数量

    for chunk in stream:
        if not getattr(chunk, 'choices', None):
            continue

        delta = chunk.choices[0].delta

        if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
            continue

        if not getattr(delta, 'reasoning_content', None) and not is_answering:
            is_answering = True

        if getattr(delta, 'reasoning_content', None):
            reasoning_content += delta.reasoning_content
        elif getattr(delta, 'content', None):
            answer_content += delta.content

        chunk_count += 1
        print(f"已接收 {chunk_count} 个数据块", end='\r')  # 实时刷新chunk数量

    return reasoning_content, answer_content

def main():
    config = read_config()
    symbols = config['symbols']
    api_key = config['openai_api_key']
    days = config['days']
    
    for i, symbol in enumerate(symbols):
        df = read_stock_data(symbol)
        if df is not None:
            analysis_output = print_stock_analysis(symbol, df, n=days)
            print(f"分析结果:\n{analysis_output}")
            reasoning, answer = get_deepseek_response(analysis_output, api_key)
            print(f"\nDeepSeek 思考过程:\n{reasoning}")
            print(f"\nDeepSeek 回复:\n{answer}")
            
            # 收集输出内容
            output_content = f"分析结果:\n{analysis_output}\n\nDeepSeek 思考过程:\n{reasoning}\n\nDeepSeek 回复:\n{answer}\n"
            
            # 输出到文本文件
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"{symbol.replace('.', '_')}_{date_str}_report.txt"
            with open(filename, 'w', encoding='utf8') as f:
                f.write(output_content)
            
            if i < len(symbols) - 1:
                print("=" * 20)

if __name__ == "__main__":
    main()

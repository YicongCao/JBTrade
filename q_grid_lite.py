import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils import get_csv_filename, read_config
from update_k_line_hist import fetch_and_save_k_line_data

GRID_CONFIG = {
    'initial_cash': 1000000,
    'grid_pct': 0.04, 
    'grid_size': 4,   
    'per_grid_cash': 50000,
    'max_position_ratio': 0.3,
    'adaptive_grid_N': 60,
}

# ===================== 网格交易核心类 =====================
class GridTraderLite:
    def __init__(self, stock_data_dict, config):
        self.config = config
        self.symbols = list(stock_data_dict.keys())
        self.data_dict = stock_data_dict
        self.initial_cash = config['initial_cash']
        self.cash = self.initial_cash
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.asset_history = []  # 记录每日资产
        self.grid_levels = {}
        self.setup_grids()

    def setup_grids(self, mid_price_dict=None):
        # 支持传入每只股票的中轴价，否则用首日收盘
        for symbol, df in self.data_dict.items():
            if mid_price_dict and symbol in mid_price_dict:
                mid_price = mid_price_dict[symbol]
            else:
                mid_price = df['close'].iloc[0]
            grid_pct = self.config['grid_pct']
            grid_size = self.config['grid_size']
            levels = [mid_price * (1 + grid_pct * (i - grid_size)) for i in range(2 * grid_size + 1)]
            self.grid_levels[symbol] = sorted(levels)

    def run_backtest(self):
        trades_log = []  # 记录每日操作
        all_dates = sorted(set(sum([list(df['time_key'].dt.strftime('%Y-%m-%d')) for df in self.data_dict.values()], [])))
        daily_report = []
        self.trade_stats = {symbol: {'BUY': 0, 'SELL': 0, 'NO_TRADE': 0} for symbol in self.symbols}
        N = self.config.get('adaptive_grid_N', 5)
        for date in all_dates:
            # 用最近N日均价作为自适应网格中轴
            mid_price_dict = {}
            for symbol, df in self.data_dict.items():
                df_sorted = df.sort_values('time_key')
                df_recent = df_sorted[df_sorted['time_key'].dt.strftime('%Y-%m-%d') <= date].tail(N)
                if not df_recent.empty:
                    mid_price_dict[symbol] = df_recent['close'].mean()
                else:
                    mid_price_dict[symbol] = df_sorted['close'].iloc[0]
            self.setup_grids(mid_price_dict)
            price_dict = {symbol: df[df['time_key'].dt.strftime('%Y-%m-%d')==date]['close'].values[0] if not df[df['time_key'].dt.strftime('%Y-%m-%d')==date].empty else None for symbol, df in self.data_dict.items()}
            total_value = self.cash
            for symbol in self.symbols:
                if price_dict[symbol] is not None:
                    total_value += self.positions[symbol] * price_dict[symbol]
            self.asset_history.append({'date': date, 'total_value': total_value, 'cash': self.cash, 'positions': self.positions.copy()})
            for symbol in self.symbols:
                price = price_dict[symbol]
                did_trade = False
                trade_type = 'NO_TRADE'
                trade_qty = 0
                trigger_price = None
                if price is None:
                    self.trade_stats[symbol]['NO_TRADE'] += 1
                    trades_log.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'NO_TRADE',
                        'price': '',
                        'quantity': '',
                        'trigger_price': ''
                    })
                    continue
                cur_value = self.positions[symbol] * price
                max_value = self.config['max_position_ratio'] * total_value
                # 买入
                if cur_value < max_value and self.cash >= self.config['per_grid_cash']:
                    for level in self.grid_levels[symbol]:
                        if price <= level:
                            qty = int(self.config['per_grid_cash'] // price)
                            if qty > 0:
                                self.cash -= qty * price
                                self.positions[symbol] += qty
                                self.trade_stats[symbol]['BUY'] += 1
                                did_trade = True
                                trade_type = 'BUY'
                                trade_qty = qty
                                trigger_price = level
                            break
                # 卖出
                if cur_value > 0:
                    for level in self.grid_levels[symbol]:
                        if price >= level:
                            qty = min(self.positions[symbol], int(self.config['per_grid_cash'] // price))
                            if qty > 0:
                                self.cash += qty * price
                                self.positions[symbol] -= qty
                                self.trade_stats[symbol]['SELL'] += 1
                                did_trade = True
                                trade_type = 'SELL'
                                trade_qty = qty
                                trigger_price = level
                            break
                if not did_trade:
                    self.trade_stats[symbol]['NO_TRADE'] += 1
                trades_log.append({
                    'date': date,
                    'symbol': symbol,
                    'action': trade_type,
                    'price': price,
                    'quantity': trade_qty if did_trade else '',
                    'trigger_price': trigger_price if did_trade else ''
                })
            daily_report.append(self._gen_report(date, price_dict))
        # 写入 trades.csv
        import csv
        with open('trades.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'symbol', 'action', 'price', 'quantity', 'trigger_price'])
            writer.writeheader()
            for row in trades_log:
                writer.writerow(row)
        return daily_report

    def _gen_report(self, date, price_dict):
        report = {'date': date, 'cash': self.cash, 'positions': {}, 'total_value': 0}
        total_value = self.cash
        for symbol in self.symbols:
            price = price_dict.get(symbol, 0) or 0
            qty = self.positions[symbol]
            value = qty * price
            report['positions'][symbol] = {'quantity': qty, 'price': price, 'value': value}
            total_value += value
        report['total_value'] = total_value
        return report

    def monthly_report(self):
        df = pd.DataFrame(self.asset_history)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        return df.groupby('month')['total_value'].last()

    def yearly_report(self):
        df = pd.DataFrame(self.asset_history)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        return df.groupby('year')['total_value'].last()

# ===================== 今日网格操作建议 =====================
def grid_signal_today(stock_data_dict, config):
    results = []
    today_str = datetime.now().strftime('%Y-%m-%d')
    for symbol, df in stock_data_dict.items():
        if len(df) < 3:
            results.append({'symbol': symbol, 'alert': '数据不足3天，无法生成信号'})
            continue
        df = df.sort_values('time_key')
        last_row = df.iloc[-1]
        price = last_row['close']
        # 用最近N日均价作为自适应网格中轴
        N = config.get('adaptive_grid_N', 5)
        if len(df) >= N:
            mid_price = df['close'].iloc[-N:].mean()
        else:
            mid_price = df['close'].iloc[0]
        grid_pct = config['grid_pct']
        grid_size = config['grid_size']
        grid_levels = [mid_price * (1 + grid_pct * (i - grid_size)) for i in range(2 * grid_size + 1)]
        grid_levels = sorted(grid_levels)
        action = 'NO_TRADE'
        reason = ''
        op_type = ''
        trigger_price = None
        for level in grid_levels:
            if price <= level:
                action = 'TRADE'
                op_type = 'BUY'
                reason = '触发买入网格'
                trigger_price = level
                break
        if action == 'NO_TRADE':
            for level in grid_levels:
                if price >= level:
                    action = 'TRADE'
                    op_type = 'SELL'
                    reason = '触发卖出网格'
                    trigger_price = level
                    break
        if action == 'NO_TRADE':
            reason = '未触发任何网格'
        # 未来三次BUY: 低于当前价的最接近的3个网格线
        buy_prices = [level for level in grid_levels if level < price]
        buy_prices = buy_prices[:3] if len(buy_prices) >= 3 else buy_prices
        # 未来三次SELL: 高于当前价的最接近的3个网格线
        sell_prices = [level for level in grid_levels if level > price]
        sell_prices = sell_prices[:3] if len(sell_prices) >= 3 else sell_prices
        results.append({
            'symbol': symbol,
            'action': op_type if action == 'TRADE' else 'NO_TRADE',
            'reason': reason,
            'price': price,
            'trigger_price': trigger_price,
            'future_buy_prices': buy_prices,
            'future_sell_prices': sell_prices,
            'date': today_str
        })
    return results

# ===================== 回测封装 =====================
def run_backtest_and_report(stock_data_dict, config):
    trader = GridTraderLite(stock_data_dict, config)
    daily_report = trader.run_backtest()
    print("\n===== 各股票操作次数统计 =====")
    for symbol, stats in trader.trade_stats.items():
        print(f"{symbol}: BUY={stats['BUY']} SELL={stats['SELL']} NO_TRADE={stats['NO_TRADE']}")
    print("\n===== 年报 =====")
    print(trader.yearly_report())
    print("\n===== 月报 =====")
    print(trader.monthly_report())
    print("\n===== 日报（最后5天） =====")
    for r in daily_report[-5:]:
        print(r)

# ===================== 推送到企业微信 =====================
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

# ===================== 主流程 =====================
if __name__ == "__main__":
    def gen_and_push_signals(stock_data_dict, grid_config, global_config, tmp_json_file, wxwork_key):
        today_signals = grid_signal_today(stock_data_dict, grid_config)
        with open(tmp_json_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(today_signals, f, ensure_ascii=False, indent=2)
        all_msgs = []
        # 构建 symbol -> name 的映射
        symbol_name_map = {s: n for s, n in zip(global_config.get('symbols', []), global_config.get('names', []))} if global_config.get('names') else {}
        for s in today_signals:
            buy_str = ', '.join([f'{x:.2f}' for x in s.get('future_buy_prices', [])]) if s.get('future_buy_prices') else '-'
            sell_str = ', '.join([f'{x:.2f}' for x in s.get('future_sell_prices', [])]) if s.get('future_sell_prices') else '-'
            trigger_price_str = f"{s['trigger_price']:.2f}" if s.get('trigger_price') not in (None, '') else ''
            symbol = s['symbol']
            name = symbol_name_map.get(symbol, symbol)
            msg = f"**{name}({symbol})**\n操作: {s['action']}\n价格: {s['price']}\n原因: {s['reason']}\n触发网格价: {trigger_price_str}\n未来三次BUY价: {buy_str}\n未来三次SELL价: {sell_str}"
            print(msg)
            all_msgs.append(msg)
        if wxwork_key:
            full_msg = '\n\n'.join(all_msgs)
            def split_by_bytes(s, max_bytes=1950):
                res = []
                cur = ''
                for line in s.split('\n'):
                    if len((cur + '\n' + line).encode('utf-8')) > max_bytes:
                        res.append(cur)
                        cur = line
                    else:
                        if cur:
                            cur += '\n' + line
                        else:
                            cur = line
                if cur:
                    res.append(cur)
                return res
            msg_chunks = split_by_bytes(full_msg, 1950)
            for chunk in msg_chunks:
                push_to_wxwork(chunk, wxwork_key)
    # 更新股价数据
    fetch_and_save_k_line_data()
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

    # 通过 config 控制运行模式
    mode = global_config.get('mode', 'backtest')  # 'backtest' or 'signal'
    if mode == 'backtest':
        run_backtest_and_report(stock_data_dict, GRID_CONFIG)
    elif mode == 'signal':
        import json
        wxwork_key = global_config.get('wxwork_webhook_key')
        tmp_json_file = 'q_grid_lite_tmp.json'
        today_str = datetime.now().strftime('%Y-%m-%d')
        # 检查所有股票最新数据日期是否为今天
        latest_dates = []
        for symbol, df in stock_data_dict.items():
            if not df.empty:
                latest_date = df.sort_values('time_key').iloc[-1]['time_key'].strftime('%Y-%m-%d')
                latest_dates.append(latest_date)
        if not latest_dates or any(d != today_str for d in latest_dates):
            print(f"非交易日或数据未更新到今日({today_str})，程序自动退出。")
            import sys
            sys.exit(0)
        # 检查是否有临时信号文件
        if os.path.exists(tmp_json_file):
            with open(tmp_json_file, 'r', encoding='utf-8') as f:
                prev_signals = json.load(f)
            # 检查临时信号文件日期
            prev_date = prev_signals[0]['date'] if prev_signals and 'date' in prev_signals[0] else None
            if prev_date != today_str:
                # 日期不是今天，重新生成今日信号并推送
                gen_and_push_signals(stock_data_dict, GRID_CONFIG, global_config, tmp_json_file, wxwork_key)
            else:
                # 日期是今天，检查是否触及未来三次SELL/BUY价
                triggered = False
                for symbol in global_config['symbols']:
                    csv_file = get_csv_filename(symbol)
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        df['time_key'] = pd.to_datetime(df['time_key'])
                        latest_row = df.sort_values('time_key').iloc[-1]
                        latest_price = latest_row['close']
                        prev = next((x for x in prev_signals if x['symbol'] == symbol), None)
                        if prev:
                            sell_hit = [x for x in prev.get('future_sell_prices', []) if latest_price >= x]
                            buy_hit = [x for x in prev.get('future_buy_prices', []) if latest_price <= x]
                            if sell_hit or buy_hit:
                                buy_str = ', '.join([f'{x:.2f}' for x in prev.get('future_buy_prices', [])]) if prev.get('future_buy_prices') else '-'
                                sell_str = ', '.join([f'{x:.2f}' for x in prev.get('future_sell_prices', [])]) if prev.get('future_sell_prices') else '-'
                                msg = f"**{symbol}**\n最新价: {latest_price}\n未来三次BUY价: {buy_str}\n未来三次SELL价: {sell_str}"
                                if buy_hit:
                                    msg += f"\n已触及BUY价: {', '.join([f'{x:.2f}' for x in buy_hit])}"
                                if sell_hit:
                                    msg += f"\n已触及SELL价: {', '.join([f'{x:.2f}' for x in sell_hit])}"
                                print(msg)
                                if wxwork_key:
                                    push_to_wxwork(msg, wxwork_key)
                                triggered = True
                if triggered:
                    # 删除临时文件并生成新信号
                    os.remove(tmp_json_file)
                    gen_and_push_signals(stock_data_dict, GRID_CONFIG, global_config, tmp_json_file, wxwork_key)
        else:
            # 今日网格操作建议并推送到企业微信
            gen_and_push_signals(stock_data_dict, GRID_CONFIG, global_config, tmp_json_file, wxwork_key)

import os
import json
import pandas as pd
import numpy as np
from utils import get_csv_filename, read_config
from datetime import datetime

# ===================== 策略参数集中配置（可扩展） =====================
CONFIG = {
    'initial_cash': 1000000,      # 初始总资金
    'grid_pct': 0.02,             # 网格间距百分比
    'grid_size': 5,               # 网格数量
    'per_grid_cash': 20000,       # 每格买入资金
    'max_position_ratio': 0.3,    # 单只股票最大持仓比例（如0.3=30%）
    'order_expire_days': 1,       # 交易单有效天数
    # 可扩展更多参数
}

POSITION_FILE = 'position.json'
ORDERS_FILE = 'orders.json'

# ===================== 仓位管理类 =====================
class Position:
    def __init__(self, positions=None, cash=None, initial_cash=None):
        self.positions = positions if positions is not None else {}
        self.cash = cash if cash is not None else (initial_cash if initial_cash is not None else 0)

    @classmethod
    def load(cls, filename, initial_cash):
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(data.get('positions', {}), data.get('cash', initial_cash), initial_cash)
        else:
            return cls({}, initial_cash, initial_cash)

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({'positions': self.positions, 'cash': self.cash}, f, ensure_ascii=False, indent=2)

    def total_value(self, price_dict):
        value = self.cash
        for symbol, pos in self.positions.items():
            value += pos.get('quantity', 0) * price_dict.get(symbol, 0)
        return value

    def position_ratio(self, symbol, price_dict):
        total = self.total_value(price_dict)
        if total == 0:
            return 0
        return (self.positions.get(symbol, {}).get('quantity', 0) * price_dict.get(symbol, 0)) / total

# ===================== 交易单结构 =====================
# order = {
#   'date': '2025-06-16',
#   'symbol': 'HK_00700',
#   'type': 'BUY'/'SELL',
#   'price': 300.0,
#   'quantity': 100,
#   'status': 'PENDING'/'FILLED'/'CANCELLED',
#   'expire_date': '2025-06-17',
#   ...
# }

def load_orders(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_orders(filename, orders):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)

# ===================== 网格交易单生成 =====================
def generate_grid_orders(date, stock_data_dict, position: Position, config):
    """生成当日所有股票的条件交易单，保存到 orders.json"""
    orders = []
    price_dict = {symbol: df[df['time_key']==date]['close'].values[0] if not df[df['time_key']==date].empty else None for symbol, df in stock_data_dict.items()}
    total_value = position.total_value(price_dict)
    for symbol, df in stock_data_dict.items():
        if price_dict[symbol] is None:
            continue
        # 计算最大可持仓市值
        max_value = config['max_position_ratio'] * total_value
        cur_value = position.positions.get(symbol, {}).get('quantity', 0) * price_dict[symbol]
        # 生成买单
        if cur_value < max_value and position.cash >= config['per_grid_cash']:
            grid_levels = [price_dict[symbol] * (1 + config['grid_pct'] * (i - config['grid_size'])) for i in range(2 * config['grid_size'] + 1)]
            for level in sorted(grid_levels):
                if price_dict[symbol] <= level:
                    qty = int(config['per_grid_cash'] // price_dict[symbol])
                    if qty > 0:
                        orders.append({
                            'date': date,
                            'symbol': symbol,
                            'type': 'BUY',
                            'price': price_dict[symbol],
                            'quantity': qty,
                            'status': 'PENDING',
                            'expire_date': date,
                        })
                    break
        # 生成卖单
        if cur_value > 0:
            grid_levels = [price_dict[symbol] * (1 + config['grid_pct'] * (i - config['grid_size'])) for i in range(2 * config['grid_size'] + 1)]
            for level in sorted(grid_levels):
                if price_dict[symbol] >= level:
                    qty = min(position.positions.get(symbol, {}).get('quantity', 0), int(config['per_grid_cash'] // price_dict[symbol]))
                    if qty > 0:
                        orders.append({
                            'date': date,
                            'symbol': symbol,
                            'type': 'SELL',
                            'price': price_dict[symbol],
                            'quantity': qty,
                            'status': 'PENDING',
                            'expire_date': date,
                        })
                    break
    save_orders(ORDERS_FILE, orders)
    return orders

# ===================== 检查并执行交易单 =====================
def execute_orders(date, stock_data_dict, position: Position, config):
    orders = load_orders(ORDERS_FILE)
    price_dict = {symbol: df[df['time_key']==date]['close'].values[0] if not df[df['time_key']==date].empty else None for symbol, df in stock_data_dict.items()}
    filled_orders = []
    for order in orders:
        if order['status'] != 'PENDING':
            continue
        symbol = order['symbol']
        price = price_dict.get(symbol)
        if price is None:
            continue
        # 简单模拟：市价成交
        if order['type'] == 'BUY' and position.cash >= price * order['quantity']:
            position.cash -= price * order['quantity']
            pos = position.positions.get(symbol, {'quantity': 0})
            pos['quantity'] = pos.get('quantity', 0) + order['quantity']
            position.positions[symbol] = pos
            order['status'] = 'FILLED'
            order['fill_price'] = price
            order['fill_date'] = date
            filled_orders.append(order)
        elif order['type'] == 'SELL' and position.positions.get(symbol, {}).get('quantity', 0) >= order['quantity']:
            position.cash += price * order['quantity']
            pos = position.positions.get(symbol, {'quantity': 0})
            pos['quantity'] -= order['quantity']
            if pos['quantity'] <= 0:
                del position.positions[symbol]
            else:
                position.positions[symbol] = pos
            order['status'] = 'FILLED'
            order['fill_price'] = price
            order['fill_date'] = date
            filled_orders.append(order)
    # 保存最新仓位
    position.save(POSITION_FILE)
    # 保存订单状态
    save_orders(ORDERS_FILE, orders)
    return filled_orders

# ===================== 清理未成交订单 =====================
def clean_unfilled_orders():
    orders = load_orders(ORDERS_FILE)
    new_orders = [o for o in orders if o['status'] == 'FILLED']
    save_orders(ORDERS_FILE, new_orders)

# ===================== 日报/周报/月报/年报 =====================
def report(date, stock_data_dict, position: Position):
    price_dict = {symbol: df[df['time_key']==date]['close'].values[0] if not df[df['time_key']==date].empty else 0 for symbol, df in stock_data_dict.items()}
    total_value = position.total_value(price_dict)
    print(f"\n===== {date} 日报 =====")
    print(f"现金: {position.cash:.2f}")
    print(f"总资产: {total_value:.2f}")
    print("持仓:")
    for symbol, pos in position.positions.items():
        print(f"  {symbol}: {pos['quantity']} 股, 市值: {pos['quantity']*price_dict.get(symbol,0):.2f}")
    print("-------------------")
    orders = load_orders(ORDERS_FILE)
    print("昨日成交订单:")
    for o in orders:
        if o['status'] == 'FILLED' and o['fill_date'] == date:
            print(o)
    print("昨日未成交订单:")
    for o in orders:
        if o['status'] == 'PENDING' and o['date'] == date:
            print(o)

# ===================== 主流程 =====================
if __name__ == "__main__":
    config = read_config()
    symbols = config['symbols']
    stock_data_dict = {}
    for symbol in symbols:
        csv_file = get_csv_filename(symbol)
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['time_key'] = pd.to_datetime(df['time_key'])
            stock_data_dict[symbol] = df
        else:
            print(f"未找到 {csv_file}，跳过该股票")
    # 获取当前日期
    all_dates = sorted(set(sum([list(df['time_key'].dt.strftime('%Y-%m-%d')) for df in stock_data_dict.values()], [])))
    for date in all_dates:
        # 1. 导入仓位
        position = Position.load(POSITION_FILE, CONFIG['initial_cash'])
        # 2. 生成条件单
        generate_grid_orders(date, stock_data_dict, position, CONFIG)
        # 3. 检查并执行订单
        execute_orders(date, stock_data_dict, position, CONFIG)
        # 4. 清理未成交订单
        clean_unfilled_orders()
        # 5. 日报
        report(date, stock_data_dict, position)
    # TODO: 可扩展周报、月报、年报等

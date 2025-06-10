import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import get_csv_filename, read_config

# ===================== 策略参数集中配置 =====================
CONFIG = {
    'initial_cash': 1000000,      # 初始总资金
    'grid_pct': 0.02,             # 网格间距百分比（如2%）
    'grid_size': 5,               # 网格数量（上下各5格）
    'per_grid_cash': 20000,       # 每格买入资金
    'max_stocks': 10              # 最大持仓股票数
}

# ====================== q_traces 阅读指引 =====================
# 交易原因（action 字段为 BUY/SELL，reason 字段为 trigger_grid）
# BUY, trigger_grid
# 含义：当前价格低于等于某一网格线，且资金充足，触发买入操作。

# SELL, trigger_grid
# 含义：当前价格高于等于某一网格线，且有持仓，触发卖出操作。

# 不交易原因（action 字段为 NO_TRADE，reason 字段如下）
# max_stocks_limit
# 含义：已持有的股票数量达到最大持仓数，且当前股票未持仓，无法再买入新股票。

# no_data
# 含义：当天没有该股票的数据，无法进行任何操作。

# price_above_all_grids
# 含义：当前价格高于所有网格线，没有触发任何买入信号。

# price_below_all_grids
# 含义：当前价格低于所有网格线，没有触发任何卖出信号。

# cash_not_enough
# 含义：账户现金不足以买入一格对应的股票数量，无法执行买入操作。

# no_position_to_sell
# 含义：当前没有持仓，无法执行卖出操作。

# not_trigger_grid
# 含义：价格未触及任何网格线，未触发买入或卖出信号。

class GridTradingStrategy:
    def __init__(self, stock_data_dict, config):
        self.config = config
        self.symbols = list(stock_data_dict.keys())
        self.data_dict = stock_data_dict
        self.initial_cash = config['initial_cash']
        self.cash = self.initial_cash
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.cost_basis = {symbol: 0 for symbol in self.symbols}
        self.trade_log = []
        self.portfolio_value = []
        self.grid_levels = {}
        self.setup_grids()

    def setup_grids(self):
        for symbol, df in self.data_dict.items():
            mid_price = df['close'].iloc[0]  # 以首日收盘价为中轴
            grid_pct = self.config['grid_pct']
            grid_size = self.config['grid_size']
            # 生成网格价格
            levels = [mid_price * (1 + grid_pct * (i - grid_size)) for i in range(2 * grid_size + 1)]
            self.grid_levels[symbol] = sorted(levels)

    def execute_trades(self, out_file=None, trace_file=None):
        all_dates = sorted(set(sum([list(df['time_key']) for df in self.data_dict.values()], [])))
        # 生成主交易日志文件名
        if out_file is None:
            base_name = "q_commits_"
            idx = 1
            while True:
                out_file = f"{base_name}{idx:03d}.csv"
                if not os.path.exists(out_file):
                    break
                idx += 1
        # 生成trace日志文件名
        if trace_file is None:
            base_name2 = "q_traces_"
            idx2 = 1
            while True:
                trace_file = f"{base_name2}{idx2:03d}.csv"
                if not os.path.exists(trace_file):
                    break
                idx2 += 1
        with open(out_file, 'w', encoding='utf-8') as f, open(trace_file, 'w', encoding='utf-8') as tf:
            # 主日志表头
            f.write('date,symbol,type,price,quantity,cash_after,position_after,year,pnl\n')
            # trace日志表头
            tf.write('date,symbol,action,reason,price,trigger_price,close_price\n')
            for date in all_dates:
                total_value = self.cash
                for symbol in self.symbols:
                    df = self.data_dict[symbol]
                    row = df[df['time_key'] == date]
                    if not row.empty:
                        price = row.iloc[0]['close']
                        total_value += self.positions[symbol] * price
                self.portfolio_value.append({'date': date, 'total_value': total_value, 'cash': self.cash, 'positions': self.positions.copy()})
                for symbol in self.symbols:
                    # 最大持仓数限制
                    if sum(1 for v in self.positions.values() if v > 0) >= self.config['max_stocks'] and self.positions[symbol] == 0:
                        tf.write(f"{date},{symbol},NO_TRADE,max_stocks_limit,NA,NA,NA\n")
                        continue
                    df = self.data_dict[symbol]
                    row = df[df['time_key'] == date]
                    if row.empty:
                        tf.write(f"{date},{symbol},NO_TRADE,no_data,NA,NA,NA\n")
                        continue
                    price = row.iloc[0]['close']
                    did_trade = False
                    trigger_price = None
                    # 网格买入
                    for level in self.grid_levels[symbol]:
                        if price <= level and self.cash >= self.config['per_grid_cash']:
                            qty = int(self.config['per_grid_cash'] // price)
                            if qty > 0:
                                self.buy(symbol, price, date, qty)
                                f.write(f"{date},{symbol},BUY,{price},{qty},{self.cash},{self.positions[symbol]},{pd.to_datetime(date).year},0.0\n")
                                tf.write(f"{date},{symbol},BUY,trigger_grid,{price},{level},{price}\n")
                                did_trade = True
                                trigger_price = level
                                break
                    # 网格卖出
                    for level in self.grid_levels[symbol]:
                        if price >= level and self.positions[symbol] > 0:
                            qty = min(self.positions[symbol], int(self.config['per_grid_cash'] // price))
                            if qty > 0:
                                self.sell(symbol, price, date, qty)
                                f.write(f"{date},{symbol},SELL,{price},{qty},{self.cash},{self.positions[symbol]},{pd.to_datetime(date).year},0.0\n")
                                tf.write(f"{date},{symbol},SELL,trigger_grid,{price},{level},{price}\n")
                                did_trade = True
                                trigger_price = level
                                break
                    if not did_trade:
                        # 资金不足 or 持仓不足 or 未触发网格
                        if all(price > level for level in self.grid_levels[symbol]):
                            tf.write(f"{date},{symbol},NO_TRADE,price_above_all_grids,{price},NA,{price}\n")
                        elif all(price < level for level in self.grid_levels[symbol]):
                            tf.write(f"{date},{symbol},NO_TRADE,price_below_all_grids,{price},NA,{price}\n")
                        elif self.cash < self.config['per_grid_cash']:
                            tf.write(f"{date},{symbol},NO_TRADE,cash_not_enough,{price},NA,{price}\n")
                        elif self.positions[symbol] == 0:
                            tf.write(f"{date},{symbol},NO_TRADE,no_position_to_sell,{price},NA,{price}\n")
                        else:
                            tf.write(f"{date},{symbol},NO_TRADE,not_trigger_grid,{price},NA,{price}\n")

    def buy(self, symbol, price, date, qty):
        cost = qty * price
        if self.cash >= cost and qty > 0:
            self.cash -= cost
            self.positions[symbol] += qty
            self.cost_basis[symbol] = (self.cost_basis[symbol] * (self.positions[symbol] - qty) + cost) / self.positions[symbol] if self.positions[symbol] > qty else price
            self.trade_log.append({'date': date, 'symbol': symbol, 'type': 'BUY', 'price': price, 'quantity': qty, 'cash_after': self.cash, 'position_after': self.positions[symbol]})

    def sell(self, symbol, price, date, qty):
        if self.positions[symbol] >= qty and qty > 0:
            revenue = qty * price
            self.cash += revenue
            self.positions[symbol] -= qty
            if self.positions[symbol] == 0:
                self.cost_basis[symbol] = 0
            self.trade_log.append({'date': date, 'symbol': symbol, 'type': 'SELL', 'price': price, 'quantity': qty, 'cash_after': self.cash, 'position_after': self.positions[symbol]})

    def generate_report(self):
        portfolio_df = pd.DataFrame(self.portfolio_value)
        portfolio_df.set_index('date', inplace=True)
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        cumulative_return = (portfolio_df['total_value'].iloc[-1] / self.initial_cash - 1) * 100
        annualized_return = ((portfolio_df['total_value'].iloc[-1] / self.initial_cash) ** (252/len(portfolio_df)) - 1) * 100
        portfolio_df['peak'] = portfolio_df['total_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        sharpe = portfolio_df['daily_return'].mean() / portfolio_df['daily_return'].std() * np.sqrt(252)
        trade_df = pd.DataFrame(self.trade_log)

        # ==== 新增：每只股票每年盈亏统计（含持仓浮盈） ====
        trade_df['date'] = pd.to_datetime(trade_df['date'])
        trade_df['year'] = trade_df['date'].dt.year
        trade_df['pnl'] = 0.0
        # 只统计卖出时的已实现盈亏
        for idx, row in trade_df.iterrows():
            if row['type'] == 'SELL':
                symbol = row['symbol']
                sell_qty = row['quantity']
                cost_basis = self.cost_basis[symbol] if self.cost_basis[symbol] > 0 else 0
                trade_df.at[idx, 'pnl'] = (row['price'] - cost_basis) * sell_qty
        # 统计每年每只股票的已实现盈亏
        realized_yearly_pnl = trade_df[trade_df['type']=='SELL'].groupby(['symbol','year'])['pnl'].sum().unstack(fill_value=0)
        realized_total_yearly_pnl = trade_df[trade_df['type']=='SELL'].groupby('year')['pnl'].sum()

        # 统计每年最后一天的持仓浮盈
        last_dates = trade_df.groupby(['symbol','year'])['date'].max().reset_index()
        float_pnl_dict = {}
        for _, row in last_dates.iterrows():
            symbol = row['symbol']
            year = row['year']
            last_date = row['date']
            # 找到该年最后一天的持仓和收盘价
            pos = 0
            cost_basis = 0
            # 查找最后一次交易后的持仓和成本
            trades = trade_df[(trade_df['symbol']==symbol) & (trade_df['date']<=last_date)].sort_values('date')
            for _, t in trades.iterrows():
                if t['type'] == 'BUY':
                    pos += t['quantity']
                    cost_basis = t['price']  # 近似用最后一次买入价
                elif t['type'] == 'SELL':
                    pos -= t['quantity']
            # 查找该日收盘价
            if symbol in self.data_dict:
                df = self.data_dict[symbol]
                price_row = df[df['time_key']==last_date]
                if not price_row.empty:
                    close_price = price_row.iloc[0]['close']
                    float_pnl = (close_price - cost_basis) * pos if pos > 0 else 0
                    float_pnl_dict[(symbol, year)] = float_pnl
        # 合并已实现和浮盈
        merged_yearly = realized_yearly_pnl.copy()
        for (symbol, year), float_pnl in float_pnl_dict.items():
            if year in merged_yearly.columns:
                merged_yearly.at[symbol, year] += float_pnl
            else:
                merged_yearly.at[symbol, year] = float_pnl
        merged_total = merged_yearly.sum(axis=0)

        # 单位转换为万元
        merged_yearly = merged_yearly / 10000
        merged_total = merged_total / 10000
        print("\n每只股票每年总盈亏(含持仓浮盈,单位:万元)：")
        print(merged_yearly)
        print("\n全市场每年总盈亏(含持仓浮盈,单位:万元)：")
        print(merged_total)
        print("\n" + "="*50)
        print(f"初始资金: ${self.initial_cash:,.2f}")
        print(f"最终资产: ${portfolio_df['total_value'].iloc[-1]:,.2f}")
        print(f"累计收益率: {cumulative_return:.2f}%")
        print(f"年化收益率: {annualized_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"总交易次数: {len(self.trade_log)}")
        print("="*50 + "\n")
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df['total_value'], label='Portfolio Value')
        plt.title('Portfolio Performance (Grid Trading)')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.legend()
        plt.show()
        return {
            'trade_log': trade_df,
            'portfolio': portfolio_df,
            'final_value': portfolio_df['total_value'].iloc[-1],
            'return_pct': cumulative_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe
        }

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
    strategy = GridTradingStrategy(stock_data_dict, CONFIG)
    # 先生成唯一输出文件名
    base_name = "q_commits_"
    idx = 1
    while True:
        out_file = f"{base_name}{idx:03d}.csv"
        if not os.path.exists(out_file):
            break
        idx += 1
    base_name2 = "q_traces_"
    idx2 = 1
    while True:
        trace_file = f"{base_name2}{idx2:03d}.csv"
        if not os.path.exists(trace_file):
            break
        idx2 += 1
    strategy.execute_trades(out_file=out_file, trace_file=trace_file)
    print(f"交易记录已保存到: {out_file}")
    print(f"每日追踪日志已保存到: {trace_file}")
    results = strategy.generate_report()
    print("交易记录：")
    print(results['trade_log'].tail(20))
    print("\n关键指标：")
    print(f"夏普比率: {results['sharpe']:.2f}")

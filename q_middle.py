import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from utils import get_csv_filename, read_config

# ===================== 策略参数集中配置 =====================
CONFIG = {
    'initial_cash': 1000000,  # 初始总资金
    'ma_short': 5,            # 短期均线窗口
    'ma_long': 20,            # 长期均线窗口
    'max_single_position_pct': 0.3,  # 单只股票最大持仓比例
    'max_single_trade_qty': 100,     # 单次最大买/卖股数
    'stop_loss_pct': 0.08,    # 止损线（8%）
    'take_profit_pct': 0.15,  # 止盈线（15%）
    'max_stocks': 10           # 最大持仓股票数
}

class MultiStockQuantStrategy:
    def __init__(self, stock_data_dict, config):
        self.config = config
        self.symbols = list(stock_data_dict.keys())
        self.data_dict = stock_data_dict
        self.initial_cash = config['initial_cash']
        self.cash = self.initial_cash
        self.positions = {symbol: 0 for symbol in self.symbols}  # 每只股票持仓
        self.cost_basis = {symbol: 0 for symbol in self.symbols} # 持仓成本
        self.trade_log = []
        self.portfolio_value = []
        self.setup_indicators()

    def setup_indicators(self):
        for symbol, df in self.data_dict.items():
            df['ma_short'] = df['close'].rolling(window=self.config['ma_short']).mean()
            df['ma_long'] = df['close'].rolling(window=self.config['ma_long']).mean()
            df['signal'] = 0
            df['signal'] = np.where(
                (df['ma_short'] > df['ma_long']) & (df['ma_short'].shift(1) <= df['ma_long'].shift(1)), 1, 0)
            df['signal'] = np.where(
                (df['ma_short'] < df['ma_long']) & (df['ma_short'].shift(1) >= df['ma_long'].shift(1)), -1, df['signal'])
            df.fillna(0, inplace=True)

    def execute_trades(self):
        # 合并所有股票的日期，按时间顺序遍历
        all_dates = sorted(set(sum([list(df['time_key']) for df in self.data_dict.values()], [])))
        for date in all_dates:
            # 计算当前总资产
            total_value = self.cash
            for symbol in self.symbols:
                df = self.data_dict[symbol]
                row = df[df['time_key'] == date]
                if not row.empty:
                    price = row.iloc[0]['close']
                    total_value += self.positions[symbol] * price
            self.portfolio_value.append({'date': date, 'total_value': total_value, 'cash': self.cash, 'positions': self.positions.copy()})

            # 逐股票处理信号
            for symbol in self.symbols:
                df = self.data_dict[symbol]
                row = df[df['time_key'] == date]
                if row.empty:
                    continue
                price = row.iloc[0]['close']
                signal = row.iloc[0]['signal']
                # 止损止盈判断
                if self.positions[symbol] > 0:
                    cost = self.cost_basis[symbol]
                    if price <= cost * (1 - self.config['stop_loss_pct']):
                        self.sell(symbol, price, date, reason='STOP_LOSS')
                        continue
                    if price >= cost * (1 + self.config['take_profit_pct']):
                        self.sell(symbol, price, date, reason='TAKE_PROFIT')
                        continue
                # 策略信号
                if signal == 1:
                    self.buy(symbol, price, date)
                elif signal == -1:
                    self.sell(symbol, price, date, reason='SIGNAL')

    def buy(self, symbol, price, date):
        # 仓位管理：单股最大持仓比例、最大持仓股票数
        if sum(1 for v in self.positions.values() if v > 0) >= self.config['max_stocks'] and self.positions[symbol] == 0:
            return  # 已达最大持仓股票数
        max_position_value = self.config['max_single_position_pct'] * self.initial_cash
        can_buy_value = min(self.cash, max_position_value - self.positions[symbol] * price)
        buy_qty = min(self.config['max_single_trade_qty'], int(can_buy_value // price))
        if buy_qty > 0 and price > 0:
            cost = buy_qty * price
            self.cash -= cost
            self.cost_basis[symbol] = (self.cost_basis[symbol] * self.positions[symbol] + cost) / (self.positions[symbol] + buy_qty) if self.positions[symbol] > 0 else price
            self.positions[symbol] += buy_qty
            self.trade_log.append({'date': date, 'symbol': symbol, 'type': 'BUY', 'price': price, 'quantity': buy_qty, 'cash_after': self.cash, 'position_after': self.positions[symbol]})

    def sell(self, symbol, price, date, reason='SIGNAL'):
        if self.positions[symbol] > 0:
            sell_qty = min(self.config['max_single_trade_qty'], self.positions[symbol])
            revenue = sell_qty * price
            self.cash += revenue
            self.positions[symbol] -= sell_qty
            if self.positions[symbol] == 0:
                self.cost_basis[symbol] = 0
            self.trade_log.append({'date': date, 'symbol': symbol, 'type': 'SELL', 'price': price, 'quantity': sell_qty, 'cash_after': self.cash, 'position_after': self.positions[symbol], 'reason': reason})

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
        # 统计胜率、盈亏比
        trade_df = pd.DataFrame(self.trade_log)
        win_trades = trade_df[(trade_df['type']=='SELL') & (trade_df['price']>0)]
        if not win_trades.empty:
            win_trades = win_trades.copy()
            win_trades['pnl'] = win_trades['price'] - win_trades['price'].shift(1)
            win_rate = (win_trades['pnl'] > 0).mean() * 100
            avg_win = win_trades[win_trades['pnl'] > 0]['pnl'].mean() if (win_trades['pnl'] > 0).any() else 0
            avg_loss = win_trades[win_trades['pnl'] <= 0]['pnl'].mean() if (win_trades['pnl'] <= 0).any() else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        else:
            win_rate = np.nan
            profit_factor = np.nan
        print("\n" + "="*50)
        print(f"初始资金: ${self.initial_cash:,.2f}")
        print(f"最终资产: ${portfolio_df['total_value'].iloc[-1]:,.2f}")
        print(f"累计收益率: {cumulative_return:.2f}%")
        print(f"年化收益率: {annualized_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"胜率: {win_rate:.2f}%")
        print(f"盈亏比: {profit_factor:.2f}")
        print(f"总交易次数: {len(self.trade_log)}")
        print("="*50 + "\n")
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df['total_value'], label='Portfolio Value')
        plt.title('Portfolio Performance (Multi-Stock)')
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
            'sharpe': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

if __name__ == "__main__":
    # 读取配置和数据
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
    # 回测
    strategy = MultiStockQuantStrategy(stock_data_dict, CONFIG)
    strategy.execute_trades()
    results = strategy.generate_report()
    print("交易记录：")
    print(results['trade_log'].tail(20))
    print("\n关键指标：")
    print(f"夏普比率: {results['sharpe']:.2f}")
    print(f"年化收益率: {results['annualized_return']:.2f}%")
    print(f"最大回撤: {results['max_drawdown']:.2f}%")
    print(f"胜率: {results['win_rate']:.2f}%")
    print(f"盈亏比: {results['profit_factor']:.2f}")

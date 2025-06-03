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

    def execute_trades(self):
        all_dates = sorted(set(sum([list(df['time_key']) for df in self.data_dict.values()], [])))
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
                if sum(1 for v in self.positions.values() if v > 0) >= self.config['max_stocks'] and self.positions[symbol] == 0:
                    continue
                df = self.data_dict[symbol]
                row = df[df['time_key'] == date]
                if row.empty:
                    continue
                price = row.iloc[0]['close']
                # 网格买入
                for level in self.grid_levels[symbol]:
                    if price <= level and self.cash >= self.config['per_grid_cash']:
                        qty = int(self.config['per_grid_cash'] // price)
                        if qty > 0:
                            self.buy(symbol, price, date, qty)
                    # 网格卖出
                    if price >= level and self.positions[symbol] > 0:
                        qty = min(self.positions[symbol], int(self.config['per_grid_cash'] // price))
                        if qty > 0:
                            self.sell(symbol, price, date, qty)

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
    strategy.execute_trades()
    results = strategy.generate_report()
    print("交易记录：")
    print(results['trade_log'].tail(20))
    print("\n关键指标：")
    print(f"夏普比率: {results['sharpe']:.2f}")

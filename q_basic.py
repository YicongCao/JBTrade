import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimpleQuantStrategy:
    def __init__(self, data, initial_cash=100000):
        """
        初始化量化策略
        :param data: 包含time_key,open,close,high,low,volume的数据框
        :param initial_cash: 初始资金
        """
        self.data = data.copy()
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.position = 0  # 当前持仓数量
        self.trade_log = []  # 交易记录
        self.portfolio_value = []  # 每日资产总值记录
        self.setup_indicators()
        
    def setup_indicators(self):
        """计算技术指标"""
        # 计算双均线[1,2](@ref)
        self.data['ma_short'] = self.data['close'].rolling(window=5).mean()  # 5日均线
        self.data['ma_long'] = self.data['close'].rolling(window=20).mean()  # 20日均线
        # 生成交易信号：1=买入，-1=卖出，0=持有
        self.data['signal'] = 0
        self.data['signal'] = np.where(
            (self.data['ma_short'] > self.data['ma_long']) & 
            (self.data['ma_short'].shift(1) <= self.data['ma_long'].shift(1)), 
            1, 0
        )
        self.data['signal'] = np.where(
            (self.data['ma_short'] < self.data['ma_long']) & 
            (self.data['ma_short'].shift(1) >= self.data['ma_long'].shift(1)), 
            -1, self.data['signal']
        )
        # 填充NaN值
        self.data.fillna(0, inplace=True)
    
    def execute_trades(self):
        """执行交易策略"""
        for i, row in self.data.iterrows():
            close_price = row['close']
            date = row['time_key']
            
            # 记录每日资产总值（现金 + 持仓市值）
            current_value = self.current_cash + self.position * close_price
            self.portfolio_value.append({
                'date': date,
                'total_value': current_value,
                'position': self.position,
                'price': close_price
            })
            
            # 交易信号处理
            if row['signal'] == 1:  # 买入信号
                self.buy(close_price, date)
                
            elif row['signal'] == -1:  # 卖出信号
                self.sell(close_price, date)
    
    def buy(self, price, date):
        """买入操作"""
        # 仓位管理：固定比例投资[5,6](@ref)
        max_position_value = self.current_cash * 0.3  # 单次最大仓位30%
        buy_quantity = min(100, int(max_position_value / price))  # 每次最多买100股
        
        if buy_quantity > 0 and price > 0:
            cost = buy_quantity * price
            self.current_cash -= cost
            self.position += buy_quantity
            
            # 记录交易
            self.trade_log.append({
                'date': date,
                'type': 'BUY',
                'price': price,
                'quantity': buy_quantity,
                'cash_after': self.current_cash,
                'position_after': self.position
            })
    
    def sell(self, price, date):
        """卖出操作"""
        if self.position > 0:
            sell_quantity = min(100, self.position)  # 每次最多卖100股
            revenue = sell_quantity * price
            self.current_cash += revenue
            self.position -= sell_quantity
            
            # 记录交易
            self.trade_log.append({
                'date': date,
                'type': 'SELL',
                'price': price,
                'quantity': sell_quantity,
                'cash_after': self.current_cash,
                'position_after': self.position
            })
    
    def generate_report(self):
        """生成交易报告和绩效统计"""
        # 创建资产数据框
        portfolio_df = pd.DataFrame(self.portfolio_value)
        portfolio_df.set_index('date', inplace=True)
        
        # 计算收益率
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        cumulative_return = (portfolio_df['total_value'].iloc[-1] / self.initial_cash - 1) * 100
        
        # 计算最大回撤[8](@ref)
        portfolio_df['peak'] = portfolio_df['total_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # 打印报告
        print("\n" + "="*50)
        print(f"初始资金: ${self.initial_cash:,.2f}")
        print(f"最终资产: ${portfolio_df['total_value'].iloc[-1]:,.2f}")
        print(f"累计收益率: {cumulative_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"总交易次数: {len(self.trade_log)}")
        print("="*50 + "\n")
        
        # 绘制资产曲线
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df['total_value'], label='Portfolio Value')
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # 返回结构化结果
        return {
            'trade_log': pd.DataFrame(self.trade_log),
            'portfolio': portfolio_df,
            'final_value': portfolio_df['total_value'].iloc[-1],
            'return_pct': cumulative_return,
            'max_drawdown': max_drawdown
        }

# ===================== 回测用例 =====================
if __name__ == "__main__":
    # 读取实际数据
    data = pd.read_csv("HK_00700.csv")
    # 确保 time_key 为日期类型
    data['time_key'] = pd.to_datetime(data['time_key'])

    # 运行策略
    strategy = SimpleQuantStrategy(data, initial_cash=100000)
    strategy.execute_trades()
    results = strategy.generate_report()

    # 打印交易记录
    print("交易记录：")
    print(results['trade_log'])

    # 打印关键指标
    print("\n关键指标：")
    sharpe = results['portfolio']['daily_return'].mean()/results['portfolio']['daily_return'].std()*np.sqrt(252)
    print(f"夏普比率: {sharpe:.2f}")
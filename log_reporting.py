import csv
import os
from datetime import datetime
import numpy as np

# #################### İŞLEM LOGLAMA SİSTEMİ ####################
class TradeLogger:
    def __init__(self, filename='trading_log.csv'):
        self.filename = filename
        self.ensure_file_exists()
        
    def ensure_file_exists(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'Position Type',
                    'Entry Price',
                    'Exit Price',
                    'Volume',
                    'Profit (USDT)',
                    'Return (%)',
                    'Duration',
                    'Stop Loss',
                    'Take Profit'
                ])
    
    def log_transaction(self, trade_data):
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade_data['timestamp'].isoformat(),
                    trade_data['position_type'],
                    f"{trade_data['entry_price']:.8f}",
                    f"{trade_data['exit_price']:.8f}",
                    f"{trade_data['volume']:.2f}",
                    f"{trade_data['profit']:.2f}",
                    f"{trade_data['return_pct']:.2f}%",
                    str(trade_data['duration']),
                    f"{trade_data['stop_loss']:.8f}",
                    f"{trade_data['take_profit']:.8f}"
                ])
            print(f"İşlem loglandı: {trade_data['timestamp']}")
        except Exception as e:
            print(f"Loglama hatası: {str(e)}")

# #################### PERFORMANS TAKİP SİSTEMİ ####################
class PerformanceTracker:
    def __init__(self):
        self.trade_history = []
        self.equity_curve = []
        self.starting_balance = 10000  # Başlangıç bakiyesi
        
    def log_trade(self, entry_price, exit_price, position_type, volume, timestamp):
        profit = (exit_price - entry_price) * volume if position_type == 'long' else (entry_price - exit_price) * volume
        pct_return = (profit / (entry_price * volume)) * 100
        
        trade_data = {
            'timestamp': timestamp,
            'position': position_type,
            'entry': entry_price,
            'exit': exit_price,
            'volume': volume,
            'profit': profit,
            'return_pct': pct_return
        }
        self.trade_history.append(trade_data)
        self.update_equity(profit)
        
    def update_equity(self, profit):
        current_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.starting_balance
        new_equity = current_equity + profit
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': new_equity
        })
        
    def generate_report(self):
        win_trades = [t for t in self.trade_history if t['profit'] > 0]
        loss_trades = [t for t in self.trade_history if t['profit'] <= 0]
        
        report = {
            'total_trades': len(self.trade_history),
            'win_rate': len(win_trades)/len(self.trade_history) if self.trade_history else 0,
            'avg_win': np.mean([t['return_pct'] for t in win_trades]) if win_trades else 0,
            'avg_loss': np.mean([t['return_pct'] for t in loss_trades]) if loss_trades else 0,
            'profit_factor': (sum(t['profit'] for t in win_trades)/abs(sum(t['profit'] for t in loss_trades))) if loss_trades else float('inf'),
            'max_drawdown': self.calculate_drawdown()
        }
        return report
    
    def calculate_drawdown(self):
        equity = [e['equity'] for e in self.equity_curve]
        peak = max(equity)
        trough = min(equity)
        return (peak - trough) / peak * 100

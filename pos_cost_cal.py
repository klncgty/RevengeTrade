import logging
from typing import Tuple
from config import Config

"""son güncelleme ile kümülatif biriken değerler düzelditil."""



class PositionCostCalculator:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.profit_target = Config.PROFIT_TARGET
        self.last_profit_check = 0.0
    def get_current_price(self, symbol: str) -> float:
        """
        Güncel kapanış fiyatını getirir
        """
        try:
            # Kline/Candlestick son kapanış fiyatını al
            klines = self.client.get_klines(
                symbol=f"{symbol}USDT",
                interval=Config.TIMEFRAME,
                limit=1
            )
            if klines:
                return float(klines[0][4])  # [4] indeksi close price'ı temsil eder
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 0.0   
    def get_total_balance(self, symbol: str) -> float:
        """
        Toplam coin miktarını hesaplar (serbest + kilitli)
        """
        try:
            account_info = self.client.get_account()
            asset = next((item for item in account_info['balances'] 
                        if item['asset'] == symbol), None)
            if asset:
                free = float(asset['free'])
                locked = float(asset['locked'])
                return free + locked
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting total balance: {e}")
            return 0.0
    
    def check_profit_target(self, symbol: str) -> Tuple[bool, float, float]:
        """
        Kar hedefine ulaşılıp ulaşılmadığını kontrol eder
        Returns: (hedef_aşıldı, current_profit_percentage, current_price)
        """
        try:
            avg_price, quantity, total_cost = self.get_average_entry_price(symbol)
            if not avg_price or not quantity:
                return False, 0.0, 0.0
                
            current_price = float(self.client.get_symbol_ticker(symbol=f"{symbol}USDT")['price'])
            current_value = quantity * current_price
            profit_percentage = ((current_value - total_cost) / total_cost) * 100
            
            if profit_percentage >= self.profit_target and profit_percentage > self.last_profit_check:
                self.last_profit_check = profit_percentage
                return True, profit_percentage, current_price
                
            return False, profit_percentage, current_price
            
        except Exception as e:
            self.logger.error(f"Profit check error: {e}")
            return False, 0.0, 0.0
    
    def get_average_entry_price(self, symbol: str) -> tuple:
        """
        Son alım fiyatını, mevcut coin miktarını ve toplam maliyeti hesaplar
        Returns: (son_alım_fiyatı, mevcut_miktar, toplam_maliyet)
        """
        try:
            # Get account information for current balance
            account_info = self.client.get_account()
            current_balance = next((float(asset['free']) for asset in account_info['balances'] 
                                 if asset['asset'] == symbol), 0)

            # Get last buy trade
            trades = self.client.get_my_trades(symbol=f"{symbol}USDT")
            if not trades:
                return 0, 0, 0

            # Get last buy price
            last_buy_trade = next((trade for trade in reversed(trades) 
                                 if trade['isBuyer']), None)
            
            if last_buy_trade and current_balance > 0:
                last_buy_price = float(last_buy_trade['price'])
                total_cost = current_balance * last_buy_price
                return last_buy_price, current_balance, total_cost
            
            return 0, 0, 0

        except Exception as e:
            self.logger.error(f"Error calculating position details: {e}")
            return 0, 0, 0

    def calculate_limit_sell_price(self, current_price: float, profit_percentage: float) -> float:
        """Limit satış fiyatını hesaplar"""
        return current_price * 0.999  # %0.1 buffer

    def print_position_summary(self, symbol: str):
        """Pozisyon özetini yazdırır"""
        try:
            entry_price, quantity, total_cost = self.get_average_entry_price(symbol)
            if not entry_price:
                print(f"\nNo position found for {symbol}\n")
                return

            current_price = float(self.client.get_symbol_ticker(symbol=f"{symbol}USDT")['price'])
            current_value = quantity * current_price
            pnl = current_value - total_cost
            pnl_percentage = (pnl / total_cost) * 100

            print("\n" + "="*50)
            print(f"Position Summary for {symbol}")
            print(f"Last Buy Price: {entry_price:.8f}")
            print(f"Current Price: {current_price:.8f}")
            print(f"Current Balance: {quantity:.2f}")
            print(f"Total Cost: {total_cost:.2f} USDT")
            print(f"Current Value: {current_value:.2f} USDT")
            print(f"Unrealized P/L: {pnl:.2f} USDT ({pnl_percentage:.2f}%)")
            print("="*50 + "\n")

        except Exception as e:
            self.logger.error(f"Error printing position summary: {e}")
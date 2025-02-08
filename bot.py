import ccxt
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import time
from log_reporting import TradeLogger, PerformanceTracker  
from binance.enums import *
from binance.exceptions import BinanceAPIException
from binance.client import Client
from binance.enums import *
from datetime import datetime, timedelta
import time
import logging
import colorama
from colorama import Fore, Back, Style
# metodlar
from config import Config
from dynamic_risk_manager import DynamicRiskManager
from advanced_indicators import AdvancedIndicators
from trend_strategy import TrendStrategy
from dotenv import load_dotenv
import os
load_dotenv()




# #################### İŞLEM YÜRÜTME ####################


# Initialize colorama
colorama.init()

# Configure logging with colored output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
class BinanceTradeExecutor:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret,testnet=False,requests_params={
                'timeout': 30,  
                'retries': 3     
            })
        self.strategy = TrendStrategy()
        self.logger = TradeLogger()
        self.performance_tracker = PerformanceTracker()
        self.active_position = None
        self.peak_price = None
        self.trade_count = 0
        self.total_profit = 0
        colorama.init()
    
    def print_market_status(self, data, signal):
        print("\n" + "="*50)
        print(f"{Fore.CYAN}Market Status Update - {datetime.now()}{Style.RESET_ALL}")
        print(f"Symbol: {Config.SYMBOL}USDT")
        current_close = data['close'].iloc[-1]

        print(f"Current Price: {Fore.YELLOW}{data['close'].iloc[-1]:.8f}{Style.RESET_ALL}")
        fib_support_38 = data['close'].iloc[-1] >= data['fib_38'].iloc[-1]
        fib_support_62 = data['close'].iloc[-1] >= data['fib_62'].iloc[-1]
        # MACD hesaplamaları
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        macd_momentum = (macd > macd_signal and 
                        abs(macd - macd_signal) > abs(data['macd'].iloc[-2] - data['macd_signal'].iloc[-2]))
        
        # Volume spike kontrolü
        volume_spike = data['volume_spike'].iloc[-1]
        
        # EMA koşulları
        ema_conditions = (
            abs(current_close - data['ema_50'].iloc[-1]) / data['ema_50'].iloc[-1] < 0.01 or
            abs(current_close - data['ema_200'].iloc[-1]) / data['ema_200'].iloc[-1] < 0.015 or
            data['ema_50'].iloc[-1] > data['ema_50'].iloc[-5]
        )
        
        if self.active_position:
            # Aktif pozisyon varsa satış koşullarını göster
            print("\nSell Conditions Status:")
            rsi_check = data['rsi'].iloc[-1] > 60
            macd_check = data['macd'].iloc[-1] < data['macd_signal'].iloc[-1]
            ema_check = data['close'].iloc[-1] < data['ema_50'].iloc[-1]
            volume_check = data['volume_spike'].iloc[-1]
            trend_check = self.strategy.check_trend_alignment(data) == 'bearish'
            
    
            
            
            print(f"1. RSI > 60: {Fore.GREEN if rsi_check else Fore.RED}✓ (Current: {data['rsi'].iloc[-1]:.2f}){Style.RESET_ALL}")
            print(f"2. MACD < Signal: {Fore.GREEN if macd_check else Fore.RED}✓{Style.RESET_ALL}")
            print(f"3. Price < EMA50: {Fore.GREEN if ema_check else Fore.RED}✓{Style.RESET_ALL}")
            print(f"4. Volume Spike: {Fore.GREEN if volume_check else Fore.RED}✓{Style.RESET_ALL}")
            print(f"5. Bearish Trend: {Fore.GREEN if trend_check else Fore.RED}✓{Style.RESET_ALL}")
            
            # Aktif pozisyon bilgileri
            profit = (data['close'].iloc[-1] - self.active_position['entry_price']) * self.active_position['quantity']
            print(f"\nActive Position Details:")
            print(f"Type: {Fore.GREEN}LONG{Style.RESET_ALL}")
            print(f"Entry Price: {self.active_position['entry_price']:.8f}")
            print(f"Current P/L: {Fore.GREEN if profit > 0 else Fore.RED}{profit:.2f} USDT{Style.RESET_ALL}")
            print(f"Stop Loss: {self.active_position['stop_loss']:.8f}")
            print(f"Take Profit: {self.active_position['take_profit']:.8f}")
        else:
            # Aktif pozisyon yoksa alım koşullarını göster
            print("\nBuy Conditions Status:")
            rsi_check = data['rsi'].iloc[-1] < 55  # RSI eşiğini 50 yaptık
            macd_check = data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]
            ema_check = (data['close'].iloc[-1] > data['ema_50'].iloc[-1] or 
                        data['close'].iloc[-1] > data['ema_200'].iloc[-1])
            volume_check = data['volume_spike'].iloc[-1]
            
            print(f"1. RSI < 55: {Fore.GREEN if rsi_check else Fore.RED}✓ (Current: {data['rsi'].iloc[-1]:.2f}){Style.RESET_ALL}")
            print(f"2. MACD Analysis: {Fore.GREEN if (macd_momentum or volume_spike) else Fore.RED}✓ "
                                    f"(Momentum: {macd_momentum}, Volume: {volume_spike}){Style.RESET_ALL}")
            print(f"3. EMA Conditions: {Fore.GREEN if ema_conditions else Fore.RED}✓ "
                        f"(EMA50 Dist: {abs(current_close - data['ema_50'].iloc[-1]) / data['ema_50'].iloc[-1]:.3f}, "
                        f"EMA200 Dist: {abs(current_close - data['ema_200'].iloc[-1]) / data['ema_200'].iloc[-1]:.3f}){Style.RESET_ALL}")
            print(f"4. Volume Analysis: {Fore.GREEN if volume_spike else Fore.RED}✓ "
                                f"(Spike: {volume_spike}){Style.RESET_ALL}")
            #print(f"2. MACD > Signal: {Fore.GREEN if macd_check else Fore.RED}✓{Style.RESET_ALL}")
            #print(f"3. Price > EMA50/200: {Fore.GREEN if ema_check else Fore.RED}✓{Style.RESET_ALL}")
            #print(f"4. Volume Spike: {Fore.GREEN if volume_check else Fore.RED}✓{Style.RESET_ALL}")
            #print(f"5. Fib 38.2 Support: {Fore.GREEN if fib_support_38 else Fore.RED}✓ ({data['fib_38'].iloc[-1]:.8f}){Style.RESET_ALL}")
            #print(f"6. Fib 61.8 Support: {Fore.GREEN if fib_support_62 else Fore.RED}✓ ({data['fib_62'].iloc[-1]:.8f}){Style.RESET_ALL}")
        
        print(f"\nSignal: {Fore.GREEN if signal == 'long' else Fore.RED if signal == 'short' else Fore.WHITE}{signal}{Style.RESET_ALL}")
        print("="*50)
    def print_trade_summary(self, trade_data):
        """Print trade summary with colors"""
        print("\n" + "*"*50)
        print(f"{Fore.CYAN}Trade Completed - {trade_data['timestamp']}{Style.RESET_ALL}")
        print(f"Position: {Fore.GREEN if trade_data['position_type'] == 'long' else Fore.RED}{trade_data['position_type']}{Style.RESET_ALL}")
        print(f"Entry Price: {trade_data['entry_price']:.8f}")
        print(f"Exit Price: {trade_data['exit_price']:.8f}")
        print(f"Volume: {trade_data['volume']:.4f}")
        print(f"Profit: {Fore.GREEN if trade_data['profit'] > 0 else Fore.RED}{trade_data['profit']:.2f} USDT{Style.RESET_ALL}")
        print(f"Return: {Fore.GREEN if trade_data['return_pct'] > 0 else Fore.RED}{trade_data['return_pct']:.2f}%{Style.RESET_ALL}")
        print(f"Duration: {trade_data['duration']}")
        print("*"*50 + "\n")
    def get_historical_data(self, symbol=Config.SYMBOL, interval=Config.TIMEFRAME, lookback="500"):
        """Fetch historical klines/candlestick data"""
        max_retries = 3
        retry_delay = 5  # saniye
        for attempt in range(max_retries):
            try:
                symbol = symbol + "USDT"
                print(f"{Fore.YELLOW}Fetching market data for {symbol}...{Style.RESET_ALL}")

                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=lookback
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignored'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                current_price = df['close'].iloc[-1]
                print(f"{Fore.GREEN}Market data fetched successfully{Style.RESET_ALL} --- Current {symbol} Price: {Fore.YELLOW}{current_price:.8f}{Style.RESET_ALL}")

                return df
            except BinanceAPIException as e:
                print(f"{Fore.RED}Binance API Error (Attempt {attempt + 1}/{max_retries}): {e}{Style.RESET_ALL}")
                if attempt < max_retries - 1:
                    print(f"{Fore.YELLOW}Retrying in {retry_delay} seconds...{Style.RESET_ALL}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logging.error(f"Failed to fetch data after {max_retries} attempts: {e}")
                    return None

    def calculate_position_size(self, entry_price, stop_loss):
        """
        Pozisyon büyüklüğünü hesaplar:
        1. USDT bakiyesini kontrol eder
        2. Risk miktarını hesaplar
        3. Stop loss mesafesine göre pozisyon büyüklüğünü ayarlar
        4. Minimum/maksimum işlem limitlerini kontrol eder
        """
        try:
            # Bakiye kontrolü
            account_info = self.client.get_account()
            print(f"\n{Fore.CYAN}Checking account balance...{Style.RESET_ALL}")
            
            # USDT bakiyesini bul
            usdt_balance = None
            for asset in account_info['balances']:
                if asset['asset'] == 'USDT':
                    usdt_balance = float(asset['free'])
                    print(f"Available USDT: {Fore.GREEN}{usdt_balance:.2f}{Style.RESET_ALL}")
                    break
            
            if usdt_balance is None or usdt_balance == 0:
                print(f"{Fore.RED}No USDT balance available!{Style.RESET_ALL}")
                return None
                
            # Risk hesaplama
            risk_amount = usdt_balance * Config.RISK_PER_TRADE  # Örn: %1 risk
            stop_distance = abs(entry_price - stop_loss)
            position_size = risk_amount / stop_distance
            
            # DOGE için minimum işlem miktarı (Binance TR)
            min_quantity = 100  # DOGE minimum lot
            if position_size < min_quantity:
                print(f"{Fore.YELLOW}Position size adjusted to minimum: {min_quantity}{Style.RESET_ALL}")
                position_size = min_quantity
            
            # Maksimum işlem miktarı kontrolü
            max_usdt = usdt_balance * 0.99  # Bakiyenin %99'u
            max_quantity = max_usdt / entry_price
            if position_size > max_quantity:
                position_size = max_quantity
                print(f"{Fore.YELLOW}Position size adjusted to maximum: {position_size:.2f}{Style.RESET_ALL}")
            
            # Lot büyüklüğünü yuvarla
            position_size = round(position_size, 0)  # DOGE için tam sayı
            
            print(f"\nRisk Calculation:")
            print(f"Entry Price: {entry_price:.8f}")
            print(f"Stop Loss: {stop_loss:.8f}")
            print(f"Risk Amount: {risk_amount:.2f} USDT")
            print(f"Position Size: {position_size:.2f} {Config.SYMBOL}")
            print(f"Total Value: {(position_size * entry_price):.2f} USDT")
            
            return position_size
            
        except BinanceAPIException as e:
            print(f"{Fore.RED}Binance API Error:")
            print(f"Error code: {e.code}")
            print(f"Error message: {e.message}")
            if e.code == -2015:  # Invalid API-key
                print("Please check your API permissions!")
            print(f"{Style.RESET_ALL}")
            return None
        
        except Exception as e:
            print(f"{Fore.RED}Error calculating position size: {str(e)}{Style.RESET_ALL}")
            return None

    def place_order(self, side, quantity, symbol=Config.SYMBOL, order_type=ORDER_TYPE_MARKET):
        """Place an order on Binance"""
        try:
            symbol = symbol + "USDT"
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
            return order
        except BinanceAPIException as e:
            logging.error(f"Error placing order: {e}")
            return None

    def manage_active_trade(self, current_data):
        """Manage active trade with trailing stop and dynamic targets"""
        if not self.active_position:
            return
        
        current_price = float(current_data['close'].iloc[-1])
        
        if self.active_position['type'] == 'long':
            # Stop loss kontrolü
            if current_price <= self.active_position['stop_loss']:
                print(f"{Fore.RED}Stop loss triggered at {current_price:.8f}{Style.RESET_ALL}")
                self.close_position('SELL')
                return
                
            # Trailing stop kontrolü
            if current_price > self.peak_price:
                self.peak_price = current_price
            
            trailing_stop = DynamicRiskManager.trailing_stop(
                current_price,
                self.active_position['entry_price'],
                'long',
                self.peak_price
            )
            
            if current_price <= trailing_stop:
                print(f"{Fore.YELLOW}Trailing stop triggered at {current_price:.8f}{Style.RESET_ALL}")
                self.close_position('SELL')
                    
            else:  # Short position
                if current_price < self.peak_price:
                    self.peak_price = current_price
                    
                trailing_stop = DynamicRiskManager.trailing_stop(
                    current_price,
                    self.active_position['entry_price'],
                    'short',
                    self.peak_price
                )
                
                if current_price >= trailing_stop:
                    self.close_position('BUY')

    def close_position(self, side):
        """Close active position and log results"""
        try:
            print(f"\n{Fore.YELLOW}Closing position...{Style.RESET_ALL}")
            order = self.place_order(side, self.active_position['quantity'])
            
            if order:
                exit_price = float(order['fills'][0]['price'])
                profit = (exit_price - self.active_position['entry_price']) * self.active_position['quantity']
                if self.active_position['type'] == 'short':
                    profit = -profit
                
                trade_data = {
                    'timestamp': datetime.now(),
                    'position_type': self.active_position['type'],
                    'entry_price': self.active_position['entry_price'],
                    'exit_price': exit_price,
                    'volume': self.active_position['quantity'],
                    'profit': profit,
                    'return_pct': (profit / (self.active_position['entry_price'] * self.active_position['quantity'])) * 100,
                    'duration': datetime.now() - self.active_position['entry_time'],
                    'stop_loss': self.active_position['stop_loss'],
                    'take_profit': self.active_position['take_profit']
                }
                
                self.print_trade_summary(trade_data)
                self.logger.log_transaction(trade_data)
                self.performance_tracker.log_trade(
                    self.active_position['entry_price'],
                    exit_price,
                    self.active_position['type'],
                    self.active_position['quantity'],
                    datetime.now()
                )
                
                self.trade_count += 1
                self.total_profit += profit
                
                self.active_position = None
                self.peak_price = None
                
        except Exception as e:
            logging.error(f"Error closing position: {e}")

    def execute_trade_cycle(self):
        """Main trading loop"""
        logging.info("Starting trading cycle...")
        print(f"{Fore.CYAN}Starting trading bot...{Style.RESET_ALL}")
        print(f"Trading {Config.SYMBOL}USDT on {Config.TIMEFRAME} timeframe")
        print(f"Risk per trade: {Config.RISK_PER_TRADE*100}%")
        
        while True:
            try:
                # Fetch latest market data
                data = self.get_historical_data()
                if data is None:
                    continue
                
                # Analyze market and get trading signal
                analyzed_data = self.strategy.analyze_market(data)
                signal = self.strategy.generate_signal(analyzed_data)
                self.print_market_status(analyzed_data, signal)
                # Manage existing position if any
                if self.active_position:
                    self.manage_active_trade(analyzed_data)
                    
                # Open new position if conditions are met
                elif signal in ['long', 'short']:
                    current_price = float(analyzed_data['close'].iloc[-1])
                    atr = float(analyzed_data['atr'].iloc[-1])
                    
                    # Calculate stop loss and take profit
                    stop_distance = atr * 2
                    stop_loss = current_price - stop_distance if signal == 'long' else current_price + stop_distance
                    take_profit = current_price + (stop_distance * Config.MIN_RISK_REWARD) if signal == 'long' else current_price - (stop_distance * Config.MIN_RISK_REWARD)
                    
                    # Calculate position size
                    quantity = self.calculate_position_size(current_price, stop_loss)
                    if quantity is None:
                        continue
                    
                    # Place order
                    order_side = 'BUY' if signal == 'long' else 'SELL'
                    order = self.place_order(order_side, quantity)
                    
                    if order:
                        self.active_position = {
                            'type': signal,
                            'entry_price': float(order['fills'][0]['price']),
                            'quantity': quantity,
                            'entry_time': datetime.now(),
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                        self.peak_price = self.active_position['entry_price']
                        print(f"{Fore.GREEN}Order executed successfully!{Style.RESET_ALL}")
                        print(f"Entry Price: {self.active_position['entry_price']:.8f}")
                        print(f"Quantity: {quantity:.4f}")
                        print(f"Stop Loss: {stop_loss:.8f}")
                        print(f"Take Profit: {take_profit:.8f}")
                
                # Updated countdown display for 3 minutes
                #print(f"\n{Fore.YELLOW}Next update in:{Style.RESET_ALL}")
                #total_seconds = 180  # 3 minutes = 180 seconds
                #for remaining in range(total_seconds, 0, -1):
                    #minutes = remaining // 60
                    #seconds = remaining % 60
                    #print(f"\r{Fore.YELLOW}{minutes:02d}:{seconds:02d}{Style.RESET_ALL}", end="")
                    #time.sleep(1)
                #print("\n")  # New line after countdown
                
                
                # Geri sayım ekranı
                print(f"\n{Fore.YELLOW}Next update in:{Style.RESET_ALL}")
                for remaining in range(60, 0, -1):
                    print(f"\r{Fore.YELLOW} {remaining} seconds{Style.RESET_ALL}", end="")
                    time.sleep(1)
                print("\n")  # Yeni satır
                
            except Exception as e:
                print(f"{Fore.RED}Error in trade cycle: {str(e)}{Style.RESET_ALL}")
                logging.error(f"Error in trade cycle: {e}")
                time.sleep(180)


# #################### BAŞLATMA ####################



if __name__ == "__main__":
    api_key= os.getenv("API_KEY")
    api_secret= os.getenv("API_SECRET")
       
    trader = BinanceTradeExecutor(api_key, api_secret)
    trader.execute_trade_cycle()

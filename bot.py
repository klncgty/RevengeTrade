import os
import math
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib
import colorama
from colorama import Fore, Style
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import (
    ORDER_TYPE_MARKET,
    ORDER_TYPE_LIMIT,
    TIME_IN_FORCE_GTC
)
from dotenv import load_dotenv
from typing import Dict, Optional

# Custom modules and configuration
from config import Config
from dynamic_risk_manager import DynamicRiskManager
from advanced_indicators import AdvancedIndicators
from trend_strategy import TrendStrategy
from csv_trade_logger import CSVTradeLogger
from trade_database import TradePositionManager
from initial_position_manager import InitialPositionManager
from limit_buy_order import limitBuyOrderExecutor
from ema_reject_strategy import EMARejectStrategy   
from pos_cost_cal import PositionCostCalculator
from limit_sell_order import LimitSellOrderExecutor
from loading import progress_bar
from coin_finder import CoinAnalyzer
from advanced_stoploss import AdvancedStopLossManager
from stop_loss_print import StopLossMessaging
from part_by_part import PartByPartStrategy

load_dotenv()

colorama.init()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("langchain_groq").setLevel(logging.WARNING)
class BinanceTradeExecutor:
    
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret, testnet=False, requests_params={'timeout': 30})
        self.sync_time()
        self.strategy = TrendStrategy()
        self.risk_manager = DynamicRiskManager()
        self.ema_reject = EMARejectStrategy()
        self.trade_logger = CSVTradeLogger("trade_log.csv")
        self.db = TradePositionManager()
        self.position_calculator = PositionCostCalculator(self.client)
        self.limit_sell_order = LimitSellOrderExecutor(self.client)
        self.limit_order_executer = limitBuyOrderExecutor(self.client)
        # Initialize trade tracking variables
        self.active_position = None
        self.peak_price = None
        self.trade_count = 0
        self.total_profit = 0
        self.position_status = 'ready_to_buy'
        self.initial_balance = self.get_usdt_balance()
       
        # Add missing variables for pending orders
        self.pending_sell_order = None
        self.sell_order_time = None
        
        self.last_sell_condition = None
        
        self.stop_loss_messaging = StopLossMessaging()
        self.part_strategy = PartByPartStrategy()
        # List to track successful trades
        self.successful_trades = []
          
        # Load active position at startup
        self.load_active_position()
    
    def load_active_position(self):
        """Database'den aktif pozisyonu yükler"""
        try:
            position = self.db.get_active_position(Config.SYMBOL)
            if position and isinstance(position, dict):
                # Aktif pozisyonu doğru formatta oluştur
                self.active_position = {
                    'type': 'long',  # Varsayılan olarak long
                    'entry_price': float(position['entry_price']),
                    'quantity': float(position['quantity']),
                    'entry_time': position['entry_time'] if 'entry_time' in position else datetime.now(),
                    'stop_loss': float(position['stop_loss']),
                    'take_profit': float(position['take_profit']),
                    'order_id': position['order_id']
                }
                
                # Peak price'ı entry price ile başlat
                self.peak_price = float(position['entry_price'])
                self.position_status = 'ready_to_sell'
                
                print(f"\n{Fore.CYAN}Loaded active position:{Style.RESET_ALL}")
                print(f"Entry Price: {self.active_position['entry_price']:.8f}")
                print(f"Quantity: {self.active_position['quantity']:.8f}")
                print(f"Stop Loss: {self.active_position['stop_loss']:.8f}")
                print(f"Take Profit: {self.active_position['take_profit']:.8f}")
                print(f"Order ID: {self.active_position['order_id']}")
                
                return True
            else:
                print(f"\n{Fore.YELLOW}No active position found. Starting fresh.{Style.RESET_ALL}")  
                self.active_position = None
                self.peak_price = None
                return False
                
        except Exception as e:
            logging.error(f"Error loading active position: {e}")
            self.active_position = None
            self.peak_price = None
            return False
    
    def get_usdt_balance(self) -> float:
        """Binance hesabındaki mevcut USDT bakiyesini alır."""
        try:
            account_info = self.client.get_account()
            for asset in account_info['balances']:
                if asset['asset'] == 'USDT':
                    return float(asset['free'])
            return 0.0
        except Exception as e:
            logging.error(f"Error getting USDT balance: {e}")
            return 0.0

    def get_symbol_balance(self, symbol: str) -> float:
        """Belirtilen sembolün mevcut bakiyesini alır."""
        try:
            account_info = self.client.get_account()
            asset = next((item for item in account_info['balances'] 
                        if item['asset'] == symbol), None)
            if asset:
                free = float(asset['free'])
                locked = float(asset['locked'])
                total = free + locked
                # Ondalık kısmı kaldır (aşağı yuvarla)
                return float(str(int(total)))  # veya direkt: return int(total)
            return 0.0
        except Exception as e:
            logging.error(f"Error getting {symbol} balance: {e}")
            return 0.0
        
    
    def check_position_status(self) -> None:
        """ portföy dağılımına göre pozisyon durumunu günceller.
        """
        usdt_balance = self.get_usdt_balance()
        symbol_balance = self.get_symbol_balance(Config.SYMBOL)
        ticker = self.client.get_symbol_ticker(symbol=f"{Config.SYMBOL}USDT")
        symbol_price = float(ticker['price'])
        
        symbol_value_in_usdt = symbol_balance * symbol_price
        total_portfolio = usdt_balance + symbol_value_in_usdt
        
        if total_portfolio == 0:
            return
        
        usdt_percentage = (usdt_balance / total_portfolio) * 100
        symbol_percentage = (symbol_value_in_usdt / total_portfolio) * 100

        print(f"\n{Fore.CYAN}Portfolio Distribution:{Style.RESET_ALL}")
        print(f"USDT: {Fore.YELLOW}{usdt_percentage:.2f}%{Style.RESET_ALL}")
        print(f"{Config.SYMBOL}: {Fore.YELLOW}{symbol_percentage:.2f}%{Style.RESET_ALL}")

        if usdt_percentage >= 90:
            self.position_status = 'ready_to_buy'
            print(f"{Fore.GREEN}Status: Ready to BUY{Style.RESET_ALL}")
        elif symbol_percentage >= 90:
            self.position_status = 'ready_to_sell'
            print(f"{Fore.RED}Status: Ready to SELL{Style.RESET_ALL}")

    def print_market_status(self, data, signal):
        """Piyasa durumu ve koşul detaylarını terminalde gösterir.."""
        print("\n" + "="*50)
        print(f"{Fore.CYAN}Market Status Update - {datetime.now()}{Style.RESET_ALL}")
        print(f"Symbol: {Config.SYMBOL}USDT")
        current_close = data['close'].iloc[-1]
        print(f"Current Price: {Fore.YELLOW}{current_close:.8f}{Style.RESET_ALL}")
        self.position_calculator.print_position_summary(Config.SYMBOL)

        if self.position_status == 'ready_to_sell':
            # EMA Rejection Status sadece satış için göster
            print(f"\n{Fore.CYAN}EMA-50 Rejection Status:{Style.RESET_ALL}")
            print(f"Total Rejections: {self.strategy.ema_reject.rejection_count}/{self.strategy.ema_reject.max_rejections}")
            if self.strategy.ema_reject.last_rejection_price:
                print(f"Last Rejection Price: {Fore.YELLOW}{self.strategy.ema_reject.last_rejection_price:.8f}{Style.RESET_ALL}")
                if self.strategy.ema_reject.rejection_timestamps:
                    last_rejection_time = self.strategy.ema_reject.rejection_timestamps[-1]
                    print(f"Last Rejection Time: {last_rejection_time}")
                
                
        self.strategy.print_channel_status(data)
        # Calculate MACD and volume spike status
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        macd_momentum = (macd > macd_signal and 
                         abs(macd - macd_signal) > abs(data['macd'].iloc[-2] - data['macd_signal'].iloc[-2]))
        volume_spike = data['volume_spike'].iloc[-1]
        ema_conditions = (
            abs(current_close - data['ema_50'].iloc[-1]) / data['ema_50'].iloc[-1] < 0.01 or
            abs(current_close - data['ema_200'].iloc[-1]) / data['ema_200'].iloc[-1] < 0.015 or
            data['ema_50'].iloc[-1] > data['ema_50'].iloc[-5]
        )

        # Display Buy Conditions
        """print(f"\n{Fore.CYAN}Buy Conditions Status:{Style.RESET_ALL}")
        rsi_buy_check = data['rsi'].iloc[-1] < Config.RSI_BUY
        print(f"1. RSI < {Config.RSI_BUY}: {Fore.GREEN if rsi_buy_check else Fore.RED}✓ (Current: {data['rsi'].iloc[-1]:.2f}){Style.RESET_ALL}")
        print(f"2. MACD Momentum or Volume Spike: {Fore.GREEN if (macd_momentum or volume_spike) else Fore.RED}✓{Style.RESET_ALL}")
        print(f"3. EMA Conditions: {Fore.GREEN if ema_conditions else Fore.RED}✓{Style.RESET_ALL}")"""

        # Display Sell Conditions
        if self.position_status == 'ready_to_sell':
            print(f"\n{Fore.CYAN}Sell Conditions Status:{Style.RESET_ALL}")
            rsi_sell_check = data['rsi'].iloc[-1] > Config.RSI_SELL
            macd_sell_check = data['macd'].iloc[-1] < data['macd_signal'].iloc[-1]
            ema_sell_check = data['close'].iloc[-1] < data['ema_50'].iloc[-1]
            trend_sell_check = self.strategy.check_trend_alignment(data) == 'bearish'
            print(f"1. RSI > {Config.RSI_SELL}: {Fore.GREEN if rsi_sell_check else Fore.RED}✓ (Current: {data['rsi'].iloc[-1]:.2f}){Style.RESET_ALL}")
            print(f"2. MACD < Signal: {Fore.GREEN if macd_sell_check else Fore.RED}✓{Style.RESET_ALL}")
            print(f"3. Price < EMA50: {Fore.GREEN if ema_sell_check else Fore.RED}✓{Style.RESET_ALL}")
            print(f"4. Bearish Trend: {Fore.GREEN if trend_sell_check else Fore.RED}✓{Style.RESET_ALL}")

        if self.active_position:
            profit = (current_close - self.active_position['entry_price']) * self.active_position['quantity']
            print(f"\n{Fore.CYAN}Active Position Details:{Style.RESET_ALL}")
            print(f"Type: {Fore.GREEN}LONG{Style.RESET_ALL}")
            print(f"Entry Price: {self.active_position['entry_price']:.8f}")
            print(f"Quantity: {self.active_position['quantity']:.8f}")
            print(f"Current P/L: {Fore.GREEN if profit > 0 else Fore.RED}{profit:.2f} USDT{Style.RESET_ALL}")
            print(f"Stop Loss: {self.active_position['stop_loss']:.8f}")
            print(f"Take Profit: {self.active_position['take_profit']:.8f}")
    
            if self.strategy.ema_reject.rejection_count > 0:
                print(f"\n{Fore.CYAN}EMA Rejection Status:{Style.RESET_ALL}")
                print(f"Rejection Count: {self.strategy.ema_reject.rejection_count}/{self.strategy.ema_reject.max_rejections}")
                print(f"Last Rejection Price: {self.strategy.ema_reject.last_rejection_price:.8f}")
                if self.strategy.ema_reject.pending_sell_order:
                    time_left = self.strategy.ema_reject.sell_order_expiry - datetime.now()
                    print(f"Pending Sell Order: {self.strategy.ema_reject.pending_sell_order:.8f}")
                    print(f"Order Expires In: {time_left.seconds} seconds")
        print(f"\nSignal: {Fore.GREEN if signal=='long' else Fore.RED if signal=='short' else Fore.WHITE}{signal}{Style.RESET_ALL}")
        print("="*50)

    def print_trade_summary(self, trade_data):
        """Pozisyon kapandıktan sonra ticaret özetini yazdırır."""
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
    
    def get_historical_data(self, symbol=Config.SYMBOL, interval=Config.TIMEFRAME, lookback="1000"):
        """Binance'den geçmiş mum verilerini alır ve bir DataFrame olarak döndürür."""
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                symbol_full = f"{symbol}USDT"
                print(f"{Fore.YELLOW}Fetching market data for {symbol_full}...{Style.RESET_ALL}")
                klines = self.client.get_klines(
                    symbol=symbol_full,
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
                print(f"{Fore.GREEN}Market data fetched successfully. Current {symbol_full} Price: {Fore.CYAN}{current_price:.8f}{Style.RESET_ALL}")
                
                #csv_filename = f"historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                # Save historical data to CSV file
                #df.to_csv(csv_filename, index=False)
                #print(f"{Fore.GREEN}Historical data saved to {csv_filename}{Style.RESET_ALL}")
                
                return df
            except BinanceAPIException as e:
                if e.code == -1021:
                    print(f"{Fore.YELLOW}Timestamp error detected, synchronizing time...{Style.RESET_ALL}")
                    self.sync_time()
                    continue
                print(f"{Fore.RED}Binance API Error (Attempt {attempt + 1}/{max_retries}): {e}{Style.RESET_ALL}")
                if attempt < max_retries - 1:
                    print(f"{Fore.YELLOW}Retrying in {retry_delay} seconds...{Style.RESET_ALL}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logging.error(f"Failed to fetch data after {max_retries} attempts: {e}")
                    return None

    def sync_time(self):
        """Yerel zamanı Binance sunucu zamanı ile senkronize eder."""
        try:
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            time_diff = local_time - server_time['serverTime']
            if abs(time_diff) >= 1000:
                print(f"{Fore.YELLOW}Time difference detected: {time_diff}ms{Style.RESET_ALL}")
                self.client.timestamp_offset = server_time['serverTime'] - local_time
                print(f"{Fore.GREEN}Time synchronized with Binance server.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Time synchronization error: {str(e)}{Style.RESET_ALL}")
            logging.error(f"Time sync error: {e}")

    def calculate_position_size(self, entry_price, stop_loss):
        """Mevcut USDT bakiyesine ve risk yönetimine göre pozisyon büyüklüğünü hesaplar."""
        try:
            account_info = self.client.get_account()
            print(f"\n{Fore.CYAN}Checking account balance...{Style.RESET_ALL}")
            usdt_balance = None
            for asset in account_info['balances']:
                if asset['asset'] == 'USDT':
                    usdt_balance = float(asset['free'])
                    print(f"Available USDT: {Fore.GREEN}{usdt_balance:.2f}{Style.RESET_ALL}")
                    break
            if usdt_balance is None or usdt_balance == 0:
                print(f"{Fore.RED}No USDT balance available!{Style.RESET_ALL}")
                return None
            risk_amount = usdt_balance * Config.RISK_PER_TRADE
            stop_distance = abs(entry_price - stop_loss)
            position_size = risk_amount / stop_distance
            min_quantity = 100  # Minimum lot size for coin (e.g., DOGE)
            if position_size < min_quantity:
                print(f"{Fore.YELLOW}Position size adjusted to minimum: {min_quantity}{Style.RESET_ALL}")
                position_size = min_quantity
            max_usdt = usdt_balance * 0.99
            max_quantity = max_usdt / entry_price
            if position_size > max_quantity:
                position_size = max_quantity
                print(f"{Fore.YELLOW}Position size adjusted to maximum: {position_size:.2f}{Style.RESET_ALL}")
            position_size = round(position_size, 0)
            print(f"\nRisk Calculation:")
            print(f"Entry Price: {entry_price:.8f}")
            print(f"Stop Loss: {stop_loss:.8f}")
            print(f"Risk Amount: {risk_amount:.2f} USDT")
            print(f"Position Size: {position_size:.2f} {Config.SYMBOL}")
            print(f"Total Order Value: {(position_size * entry_price):.2f} USDT")
            return position_size
        except BinanceAPIException as e:
            print(f"{Fore.RED}Binance API Error: {e.message}{Style.RESET_ALL}")
            return None
        except Exception as e:
            print(f"{Fore.RED}Error calculating position size: {str(e)}{Style.RESET_ALL}")
            return None

    def place_order(self, side, quantity, symbol=Config.SYMBOL, order_type=ORDER_TYPE_MARKET):
        """Binance üzerinde bir piyasa emri verir."""
        try:
            symbol_full = f"{symbol}USDT"
            order = self.client.create_order(
                symbol=symbol_full,
                side=side,
                type=order_type,
                quantity=quantity
            )
            return order
        except BinanceAPIException as e:
            logging.error(f"Error placing order: {e}")
            return None
    
    
    def place_limit_sell_order(self, quantity: float, price: float, symbol: str = Config.SYMBOL) -> Optional[Dict]:
        """Limit satış emri verir."""
        try:
            account_info = self.client.get_account()
            available_balance = 0
            
            for balance in account_info['balances']:
                if balance['asset'] == symbol:
                    available_balance = float(balance['free'])
                    break
            # 2. Eğer bakiye yetersizse, mevcut bakiyeyi kullan
            if available_balance < quantity:
                if available_balance <= 0:
                    print(f"{Fore.RED}No {symbol} balance available to sell{Style.RESET_ALL}")
                    return None
                print(f"{Fore.YELLOW}Adjusting sell quantity from {quantity} to {available_balance}{Style.RESET_ALL}")
                quantity = int(available_balance)
        # Quantity'yi tam sayıya yuvarla
            formatted_price = "{:.8f}".format(price)
            formatted_quantity = int(quantity)
            order = self.client.create_order(
                symbol=f"{symbol}USDT",
                side='SELL',
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,  # Added proper time in force parameter
                quantity=formatted_quantity,
                price=formatted_price
            )
            self.pending_sell_order = order
            self.sell_order_time = datetime.now()
            
            print(f"{Fore.YELLOW}Placed limit sell order at {price:.8f}{Style.RESET_ALL}")
            return order
        except BinanceAPIException as e:
            error_msg = f"Binance API error placing limit sell order: {e.message}"
            if e.code == -2010:  # Insufficient balance error
                error_msg = f"Insufficient balance for limit sell order. Required: {quantity} {symbol}"
            elif e.code == -1013:  # Invalid quantity error
                error_msg = f"Invalid quantity for limit sell order: {quantity} {symbol}"
                
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logging.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error placing limit sell order: {str(e)}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logging.error(error_msg)
            return None

    def cancel_pending_sell_order(self, symbol: str = Config.SYMBOL) -> bool:
        """Bekleyen satış emrini iptal eder"""
        try:
            if self.pending_sell_order:
                self.client.cancel_order(
                    symbol=f"{symbol}USDT",
                    orderId=self.pending_sell_order['orderId']
                )
                print(f"{Fore.RED}Cancelled pending sell order{Style.RESET_ALL}")
                self.pending_sell_order = None
                self.sell_order_time = None
                return True
        except Exception as e:
            logging.error(f"Error cancelling order: {e}")
        return False
    
    def manage_active_trade(self, current_data: pd.DataFrame) -> Optional[str]:
        """Aktif pozisyonun satılıp satılmaması gerektiğini kontrol eder ve bir "SELL" sinyali döndürür."""
        try:
            if self.position_status != 'ready_to_sell':
                return None
            if not self.active_position:
                return None

            # Retrieve current market price and the position's entry price.
            current_price = float(current_data['close'].iloc[-1])
            entry_price = self.active_position['entry_price']
            print(f"Current price: {current_price:.8f}, Entry price: {entry_price:.8f}")

            # Check for an existing pending sell order and update its status.
            if self.pending_sell_order:
                try:
                    order_status = self.client.get_order(
                        symbol=f"{Config.SYMBOL}USDT",
                        orderId=self.pending_sell_order['orderId']
                    )
                    print(f"Pending sell order status: {order_status['status']} for Order ID: {self.pending_sell_order['orderId']}")
                    if order_status['status'] in ['FILLED', 'CANCELED']:
                        self.pending_sell_order = None
                        self.sell_order_time = None
                    else:
                        # If an order is active, avoid sending a new sell signal.
                        return None
                except Exception as e_order:
                    print(f"[DEBUG] Error checking pending order status: {e_order}")
                    self.pending_sell_order = None
                    self.sell_order_time = None
            
            
            
            if Config.PART_SELL:
                part_sell = self.part_strategy.check_sell_conditions(current_price, entry_price)
                if part_sell["should_sell"]:
                    print(part_sell["message"])
                    return 'SELL'
            else:
                # ----- Profit Target Check1 -----
                target_reached, profit_percentage, _ = self.position_calculator.check_profit_target(Config.SYMBOL)
                if target_reached:
                    if profit_percentage > 3.0:
                        print(f"\033[1;32m🔥🔥🔥En yüksek kar oranına ulaşıldı🎯: {profit_percentage:.2f}% (target: {Config.PROFIT_TARGET}%)\033[0m")
                    elif profit_percentage > 1.0:
                        print(f"\033[1;32m🔥🔥Orta hedefli kar oranına ulaşıldı🎯: {profit_percentage:.2f}% (target: {Config.PROFIT_TARGET}%)\033[0m")
                    elif profit_percentage > 0.5:
                        print(f"\033[1;32m🔥Düşük hedefli kar oranına ulaşıldı🎯: {profit_percentage:.2f}% (target: {Config.PROFIT_TARGET}%)\033[0m")
                    
                    self.last_sell_condition = 'profit_target'
                    return 'SELL'  # Profit target reached
            
            # ----- Technical Sell Conditions -----
            rsi = current_data['rsi'].iloc[-1]
            macd = current_data['macd'].iloc[-1]
            macd_signal = current_data['macd_signal'].iloc[-1]
            ema_50 = current_data['ema_50'].iloc[-1]
            trend = self.strategy.check_trend_alignment(current_data)

            sell_conditions_met = (
                rsi > Config.RSI_SELL and
                macd < macd_signal and
                current_price < ema_50 and
                trend == 'bearish'
            )
            # ----- Sell conditions met -----
            if sell_conditions_met and current_price > entry_price:
                print(f"\033[92mTüm teknik satış koşulları sağlandı! SATIŞ sinyali gönderiliyor.\033[0m")
                return 'SELL'

            # ----- ema Sell Conditions -----
            if self.strategy.ema_reject.analyze_ema_rejections(current_data) and current_price > entry_price:
                print(f"\033[92mEMA reddi tespit edildi! SATIŞ sinyali gönderiliyor.\033[0m")
                return 'SELL'
            else:
                if self.strategy.ema_reject.analyze_ema_rejections(current_data) and current_price < entry_price:
                    print(f"\033[93mAlış fiyatından düşük. SATIŞ sinyali gönderilmedi.\033[0m")

            
            # ----- Stop Loss Check -----
            # Stop-loss yöneticisini başlat
            # Her fiyat güncellemesinde kontrol et
            self.stop_loss_manager = AdvancedStopLossManager(
                initial_stop_loss=self.active_position['stop_loss'],
                trailing_percentage=1.0
            )
            triggered, message = self.stop_loss_manager.check_stop_loss(
                current_price=current_price,
                current_data=current_data,
                position_entry_price=self.active_position['entry_price']
            )

            if triggered:
                print(self.stop_loss_messaging.generate_message(
                    status='STOP_LOSS',
                    current_price=current_price,
                    stop_loss_price=self.stop_loss_manager.current_stop_loss,
                    entry_price=self.active_position['entry_price'],
                    current_data=current_data,
                    reason=message
                ))
                return 'SELL'
            else:
                print(self.stop_loss_messaging.generate_message(
                    status='NO_STOP_LOSS',
                    current_price=current_price,
                    stop_loss_price=self.stop_loss_manager.current_stop_loss,
                    entry_price=self.active_position['entry_price'],
                    current_data=current_data))
            
            # ----- Trailing Stop Loss Check -----
            if self.peak_price and current_price <= self.risk_manager.trailing_stop(
                current_price=current_price,
                entry_price=entry_price,
                position_type='long',
                peak_price=self.peak_price
            ):
                print(f"\033[91mTakip eden stop tetiklendi! SATIŞ sinyali gönderiliyor.\033[0m")
                return 'SELL'

            return None

        except Exception as e:
            logging.error(f"Error in manage_active_trade: {e}")
            return None
   
    
    def close_position(self, side):
        """ Aktif pozisyonu kapatır ve ticaret detaylarını kaydeder."""
        try:
            print(f"\n{Fore.YELLOW}Closing position...{Style.RESET_ALL}")
            # Mevcut coin bakiyesini kontrol et
            actual_balance = self.get_symbol_balance(Config.SYMBOL)
            print(f"[DEBUG] Available {Config.SYMBOL} balance: {actual_balance}")
            
            if actual_balance <= 0:
                print(f"{Fore.RED}No {Config.SYMBOL} balance available to sell{Style.RESET_ALL}")
                self.active_position = None
                return
            # Pozisyon miktarını mevcut bakiyeye göre ayarla
            if self.active_position:
                if actual_balance < self.active_position['quantity']:
                    print(f"[DEBUG] Adjusting sell quantity from {self.active_position['quantity']} to {actual_balance}")
                    self.active_position['quantity'] = actual_balance
     
            if self.last_sell_condition == 'profit_target':
                ticker = self.client.get_symbol_ticker(symbol=f"{Config.SYMBOL}USDT")
                current_price = float(ticker['price'])
                # Re-run profit target check to recalculate limit price.
                target_reached, profit_percentage, _ = self.position_calculator.check_profit_target(Config.SYMBOL)
                limit_price = self.position_calculator.calculate_limit_sell_price(current_price, profit_percentage)
                print(f"[INFO] Profit target reached ({profit_percentage:.2f}%).")
                print(f"[INFO] Placing LIMIT sell order at {limit_price:.8f}.")
                order = self.limit_sell_order.place_limit_sell_order(
                    quantity=self.active_position['quantity'],
                    price=limit_price,
                    symbol=Config.SYMBOL
                )
                
            else:
                print("[INFO] Using MARKET sell order.")
                order = self.place_order(
                    side,
                    self.active_position['quantity'],
                    symbol=Config.SYMBOL,
                    order_type=ORDER_TYPE_MARKET
                )
                
            #order = self.place_order(side, self.active_position['quantity'])
            if order:
                exit_price = float(order['fills'][0]['price'])
                profit = (exit_price - self.active_position['entry_price']) * self.active_position['quantity']
                print(f"{Fore.GREEN}=== Satış Gerçekleşti! Kar Oranı: {trade_data['return_pct']:.2f}% 💰 ==={Style.RESET_ALL}")
                # EMA rejection durumunu sıfırla
                self.ema_reject.reset_rejection_count()
                self.trade_count += 1
                
                # Database'i güncelle
                self.db.update_position(
                    symbol=Config.SYMBOL,
                    profit=profit,
                    exit_reason='market_sell' if self.last_sell_condition != 'profit_target' else 'profit_target',
                    exit_price = exit_price,
                    status='closed'
    
                )
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
                self.total_profit += profit
                # Log the closed trade to CSV
                self.trade_logger.log_trade("SELL", 
                                            quantity=self.active_position['quantity'], 
                                            price=exit_price, 
                                            pnl=profit, 
                                            notes="Market sell order executed")
                print(f"[INFO] Position closed at price {exit_price:.8f}.")
                self.trade_logger.log_trade(
                    "SELL", 
                    quantity=self.active_position['quantity'], 
                    price=exit_price, 
                    pnl=profit, 
                    notes=f"{'LIMIT' if self.last_sell_condition=='profit_target' else 'MARKET'} satış emri verildi"
                )
                # Clear active position and reset last_sell_condition.
                self.active_position = None
                self.peak_price = None
                self.last_sell_condition = None
            else:
                print("[ERROR] Sell order was not placed.")
        except Exception as e:
            logging.error(f"Error closing position: {e}")
    
    def wait_for_next_timeframe(self):
        """Botun tam timeframe (5, 15, 30 dk) dilimlerine senkronize olmasını sağlar."""
        now = datetime.now()
        minutes_to_wait = Config.TIME_SYNC - (now.minute % Config.TIME_SYNC)  # Kaç dakika beklemeli?
        next_run_time = now + timedelta(minutes=minutes_to_wait)
        next_run_time = next_run_time.replace(second=0, microsecond=0)  # Tam dakikaya yuvarla
        wait_seconds = (next_run_time - now).total_seconds()
        
        print(f"Bot {next_run_time.strftime('%H:%M:%S')}'de çalışacak...")
        time.sleep(wait_seconds)
    
    def execute_trade_cycle(self):
        """Ana ticaret döngüsünü yürütür ve veritabanı güncellemelerini yapar."""
        print(f"{Fore.CYAN}Trading Bot Başlatılıyor 🔥🔥🔥{Style.RESET_ALL}")
        print(f"Trading {Config.SYMBOL}USDT on {Config.TIMEFRAME} timeframe")
        print(f"Risk per trade: {Config.RISK_PER_TRADE*100}%")
        
        while True:
            try:
                # Pozisyon durumunu kontrol et
                self.check_position_status()
                
                # Market verilerini al ve analiz et
                data = self.get_historical_data()
                data_4h = self.get_historical_data(interval=Client.KLINE_INTERVAL_4HOUR)
                data_50_mum  = self.get_historical_data(lookback="50")
                data_200_mum = self.get_historical_data(lookback="200")
                data_500_mum = self.get_historical_data(lookback="500")
                if data is None or data_4h is None or data_50_mum is None or data_200_mum is None or data_500_mum is None:
                    continue
                    
                analyzed_data = self.strategy.analyze_market(data)
                data_50_mum_analyzed = self.strategy.analyze_market(data_50_mum)
                data_200_mum_analyzed = self.strategy.analyze_market(data_200_mum)
                data_500_mum_analyzed = self.strategy.analyze_market(data_500_mum)
                signal = self.strategy.generate_signal(analyzed_data, data_4h,data_50_mum_analyzed,data_200_mum_analyzed,data_500_mum_analyzed)
                self.print_market_status(analyzed_data, signal)
                
                current_price = float(analyzed_data['close'].iloc[-1])
                has_pending_orders = False

                # Açık emirleri kontrol et
                try:
                    open_orders = self.client.get_open_orders(symbol=f"{Config.SYMBOL}USDT")
                    if open_orders:
                        has_pending_orders = True
                        print(f"\n{Fore.YELLOW}Bekleyen emirler bulundu. Yeni emir yerleştirme atlanıyor.{Style.RESET_ALL}")
                        for order in open_orders:
                            print(f"Order ID: {order['orderId']}")
                            print(f"Type: {order['type']}")
                            print(f"Side: {order['side']}")
                            print(f"Price: {order['price']}")
                            print(f"Original Quantity: {order['origQty']}")
                except Exception as e:
                    logging.error(f"Error checking open orders: {e}")
                    has_pending_orders = True

                # Alım koşullarını kontrol et
                if (self.position_status == 'ready_to_buy' 
                    and not has_pending_orders 
                    and signal in ["long", "part_buy"]):         
                    
                    # Stop loss ve take profit hesapla
                    atr = float(analyzed_data['atr'].iloc[-1])
                    stop_distance = atr * 2
                    stop_loss = current_price - stop_distance
                    take_profit = current_price + (stop_distance * Config.MIN_RISK_REWARD)
                    
                    quantity = self.calculate_position_size(current_price, stop_loss)
                    print("quantity:", quantity)
                    if quantity is None:
                        continue

                    entry_price = self.strategy.calculate_entry_price(
                        data=analyzed_data,
                        upper_line=analyzed_data['upper_channel'],
                        lower_line=analyzed_data['lower_channel'],
                        position_type='long'
                    )

                    predicted_price = self.strategy.predict_next_candle_price(analyzed_data)

                    # Limit alım emri ver
                    order = self.limit_order_executer.place_limit_buy(
                        symbol=Config.SYMBOL,
                        quantity=quantity,
                        current_price=current_price,
                        predicted_price=predicted_price if Config.PREDICT_BASED_ORDERS else None,
                        entry_price=entry_price
                    )

                    if order:
                        print(f"\n{Fore.GREEN}=== Limit Buy Order Details ==={Style.RESET_ALL}")
                        print(f"Entry Price: {entry_price:.8f}")
                        print(f"Quantity: {quantity:.4f}")
                        print(f"Stop Loss: {stop_loss:.8f}")
                        print(f"Take Profit: {take_profit:.8f}")
                        
                        # Veritabanına pending durumunda kaydet
                        position_data = {
                            'symbol': Config.SYMBOL,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'status': 'pending',  # İlk kayıt pending olarak
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'order_id': order['orderId'],
                            'entry_time': datetime.now()
                        }
                        
                        if self.db.add_position(position_data):
                            print(f"{Fore.GREEN}Pending order saved to database{Style.RESET_ALL}")
                            
                            self.active_position = {
                                'type': 'long',
                                'entry_price': entry_price,
                                'quantity': quantity,
                                'entry_time': datetime.now(),
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'order_id': order['orderId']
                            }
                            
                            self.trade_logger.log_trade(
                                "BUY", 
                                quantity=quantity, 
                                price=entry_price, 
                                notes="Limit buy order placed"
                            )

                # Bekleyen emirleri kontrol et
                filled_order = self.limit_order_executer.check_pending_buy_order(Config.SYMBOL)
                if filled_order and isinstance(filled_order, dict):
                    print(f"{Fore.GREEN}Buy order filled at {filled_order['price']}⚡{Style.RESET_ALL}")
                    
                    # Emir gerçekleşti - Status'u active yap
                    update_data = {
                        'status': 'active',
                        'entry_price': float(filled_order['price']),
                        'entry_time': datetime.now()
                    }
                    
                    if self.db.update_position(Config.SYMBOL, update_data):
                        print(f"{Fore.GREEN}Position updated to active status{Style.RESET_ALL}")
                        
                        if self.active_position:
                            self.active_position['entry_price'] = float(filled_order['price'])
                            self.peak_price = float(filled_order['price'])
                            
                            self.trade_logger.log_trade(
                                "BUY", 
                                quantity=self.active_position['quantity'], 
                                price=float(filled_order['price']), 
                                notes="Limit buy order executed"
                            )
                elif filled_order == "canceled":
                    update_data = {
                        'status': 'canceled',
                        "exit_reason": "time_out"
                    }

                    self.db.update_position(Config.SYMBOL, update_data)
                    print(f"{Fore.YELLOW}Position updated to canceled status ⚠️{Style.RESET_ALL}")
                # Aktif pozisyonları yönet
                if self.active_position and not has_pending_orders and Config.STOP_SELL==False:
                    
                    action = self.manage_active_trade(analyzed_data)
                    if action == 'SELL'or signal == "part_sell":
                        self.close_position('SELL')
                elif not has_pending_orders:
                    coin_balance = self.get_symbol_balance(Config.SYMBOL)
                    if coin_balance > 0:
                        print("[DEBUG] Active position not found, but coin balance exists. Initiating sale.")
                        action = self.manage_active_trade(analyzed_data)
                        if action == 'SELL':
                            self.close_position('SELL')
                    else:
                        print("[DEBUG] No active position and no coin balance found.")

                # Sonraki döngü için bekle
                minute = int(Config.LENGTH_BAR / 60)
                print(f"\n{Fore.YELLOW}Gelecek güncellemeye:{minute} dakika var. {Style.RESET_ALL}")
                print(f"\n{Fore.BLUE}🔥Toplam trade sayısı:⚡ {self.trade_count} ⚡  {Style.RESET_ALL}")

                #progress_bar(Config.LENGTH_BAR)
                for remaining in range(Config.LENGTH_BAR, 0, -1):
                    minutes, seconds = divmod(remaining, 60)
                    print(f"\r{Fore.YELLOW}{minutes:02d}:{seconds:02d} minutes{Style.RESET_ALL}", end="")
                    time.sleep(1)
                    
                print("\n")

            except Exception as e:
                print(f"{Fore.RED}Error in trade cycle: {str(e)}{Style.RESET_ALL}")
                logging.error(f"Error in trade cycle: {e}")
                time.sleep(Config.LENGTH_BAR)
        
if __name__ == "__main__":
    API_KEY = os.getenv("API_KEY_")
    API_SECRET = os.getenv("API_SECRET_")
    """coins = ["SHIB", "BEAMX", "NOT", "SLP", "HOT","BTTC","PEPE","XEC","SPELL","COS","RVN"]
    analyzer = CoinAnalyzer(API_KEY, API_SECRET, coins)
    try:
        response = analyzer.analyze_trends()
        print("\n")
        print("Analiz sonucu:", "\033[92m", response.content, "\033[0m")
    except Exception as e:
        print("An error occurred:", e)
        print("====================================")
        print("\n")
    print("Coinler:", coins)
    print("\n")
    print("====================================")
    print("\n")
    selected_coin = input("Bu  analizden sonra hangi coin ile işlem yapmak istersiniz? ")
    
    Config.SYMBOL = selected_coin.upper()"""
    
   
    """position_manager = InitialPositionManager(
    client=trader.client,
    db=trader.db,
    get_symbol_balance=trader.get_symbol_balance
)

# Manuel pozisyon ekle
    position_manager.add_manual_position(
        symbol='SHIB',
        entry_price=0.00001578,order_id ="MANUAL")"""
    trader = BinanceTradeExecutor(API_KEY, API_SECRET)
    trader.wait_for_next_timeframe()
    trader.execute_trade_cycle()
   
   
    

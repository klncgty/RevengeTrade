

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
from binance.enums import ORDER_TYPE_MARKET

from dotenv import load_dotenv

# Custom modules and configuration
from config import Config
from dynamic_risk_manager import DynamicRiskManager
from advanced_indicators import AdvancedIndicators
from trend_strategy import TrendStrategy
from log_reporting import TradeLogger, PerformanceTracker
from csv_trade_logger import CSVTradeLogger

# Load environment variables
load_dotenv()

# Initialize colorama for colored terminal output
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class BinanceTradeExecutor:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret, testnet=False, requests_params={'timeout': 30})
        self.sync_time()
        self.strategy = TrendStrategy()
        self.logger = TradeLogger()
        self.performance_tracker = PerformanceTracker()
        self.trade_logger = CSVTradeLogger("trade_log.csv")  # CSV trade logger integrated
        
        self.active_position = None
        self.peak_price = None
        self.trade_count = 0
        self.total_profit = 0
        self.position_status = 'ready_to_buy'  # Either 'ready_to_buy' or 'ready_to_sell'
        self.initial_balance = self.get_usdt_balance()
    
    def get_usdt_balance(self) -> float:
        """Fetch available USDT balance from Binance account."""
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
        """Fetch available symbol balance (e.g., SHIB) from Binance account."""
        try:
            account_info = self.client.get_account()
            for asset in account_info['balances']:
                if asset['asset'] == symbol:
                    return float(asset['free'])
            return 0.0
        except Exception as e:
            logging.error(f"Error getting {symbol} balance: {e}")
            return 0.0

    def check_position_status(self) -> None:
        """
        Update position_status based on current portfolio distribution.
        If USDT makes up 90% or more, set as ready_to_buy.
        If the coin (Config.SYMBOL) accounts for 90% or more in value, set as ready_to_sell.
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
        """Display market status and condition details on the terminal."""
        print("\n" + "="*50)
        print(f"{Fore.CYAN}Market Status Update - {datetime.now()}{Style.RESET_ALL}")
        print(f"Symbol: {Config.SYMBOL}USDT")
        current_close = data['close'].iloc[-1]
        print(f"Current Price: {Fore.YELLOW}{current_close:.8f}{Style.RESET_ALL}")

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
        print(f"\n{Fore.CYAN}Buy Conditions Status:{Style.RESET_ALL}")
        rsi_buy_check = data['rsi'].iloc[-1] < Config.RSI_BUY
        print(f"1. RSI < {Config.RSI_BUY}: {Fore.GREEN if rsi_buy_check else Fore.RED}✓ (Current: {data['rsi'].iloc[-1]:.2f}){Style.RESET_ALL}")
        print(f"2. MACD Momentum or Volume Spike: {Fore.GREEN if (macd_momentum or volume_spike) else Fore.RED}✓{Style.RESET_ALL}")
        print(f"3. EMA Conditions: {Fore.GREEN if ema_conditions else Fore.RED}✓{Style.RESET_ALL}")

        # Display Sell Conditions
        print(f"\n{Fore.CYAN}Sell Conditions Status:{Style.RESET_ALL}")
        rsi_sell_check = data['rsi'].iloc[-1] > 60
        macd_sell_check = data['macd'].iloc[-1] < data['macd_signal'].iloc[-1]
        ema_sell_check = data['close'].iloc[-1] < data['ema_50'].iloc[-1]
        trend_sell_check = self.strategy.check_trend_alignment(data) == 'bearish'
        print(f"1. RSI > 60: {Fore.GREEN if rsi_sell_check else Fore.RED}✓ (Current: {data['rsi'].iloc[-1]:.2f}){Style.RESET_ALL}")
        print(f"2. MACD < Signal: {Fore.GREEN if macd_sell_check else Fore.RED}✓{Style.RESET_ALL}")
        print(f"3. Price < EMA50: {Fore.GREEN if ema_sell_check else Fore.RED}✓{Style.RESET_ALL}")
        print(f"4. Bearish Trend: {Fore.GREEN if trend_sell_check else Fore.RED}✓{Style.RESET_ALL}")

        if self.active_position:
            profit = (current_close - self.active_position['entry_price']) * self.active_position['quantity']
            print(f"\n{Fore.CYAN}Active Position Details:{Style.RESET_ALL}")
            print(f"Type: {Fore.GREEN}LONG{Style.RESET_ALL}")
            print(f"Entry Price: {self.active_position['entry_price']:.8f}")
            print(f"Current P/L: {Fore.GREEN if profit > 0 else Fore.RED}{profit:.2f} USDT{Style.RESET_ALL}")
            print(f"Stop Loss: {self.active_position['stop_loss']:.8f}")
            print(f"Take Profit: {self.active_position['take_profit']:.8f}")

        print(f"\nSignal: {Fore.GREEN if signal=='long' else Fore.RED if signal=='short' else Fore.WHITE}{signal}{Style.RESET_ALL}")
        print("="*50)

    def print_trade_summary(self, trade_data):
        """Print trade summary details after closing a position."""
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
        """Fetch historical candlestick data from Binance and return as a DataFrame."""
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
                print(f"{Fore.GREEN}Market data fetched successfully. Current {symbol_full} Price: {Fore.YELLOW}{current_price:.8f}{Style.RESET_ALL}")
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
        """Synchronize local time with Binance server time."""
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
        """
        Calculate the position size based on available USDT balance and risk management.
        Adjusts to minimum quantity for the coin if necessary.
        """
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
        """Place a market order on Binance."""
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

    def manage_active_trade(self, current_data):
        """
        Manage an active trade by checking for stop loss or trailing stop triggers.
        Uses DynamicRiskManager's trailing_stop to calculate dynamic exit levels.
        Now includes a sell condition that triggers if the current price drops below EMA50
        OR if the current price has reached at least 5% profit relative to the entry price.
        """
        if not self.active_position:
            return

        current_price = float(current_data['close'].iloc[-1])
        if self.active_position['type'] == 'long':
            if current_price <= self.active_position['stop_loss']:
                print(f"{Fore.RED}Stop loss triggered at {current_price:.8f}{Style.RESET_ALL}")
                self.close_position('SELL')
                return
            ema50_value = float(current_data['ema_50'].iloc[-1])
            profit_target = self.active_position['entry_price'] * 1.05
            #########
            if current_price < ema50_value or current_price >= profit_target:
                print(f"{Fore.RED}Sell triggered: current_price {current_price:.8f} "
                      f"(EMA50: {ema50_value:.8f} OR Profit Target (5%) reached: {profit_target:.8f}){Style.RESET_ALL}")
                sell_limit_price = current_data['high'].max()
                self.close_position('SELL', sell_limit_price)
                return
            if current_price < ema50_value or current_price >= profit_target:
                print(f"{Fore.RED}Sell triggered: current_price {current_price:.8f} "
                      f"(EMA50: {ema50_value:.8f} OR Profit Target (5%) reached: {profit_target:.8f}){Style.RESET_ALL}")
                sell_limit_price = current_data['high'].max()
                self.close_position('SELL', sell_limit_price)
                return
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
        else:
            if current_price < self.peak_price:
                self.peak_price = current_price
            trailing_stop = DynamicRiskManager.trailing_stop(
                current_price,
                self.active_position['entry_price'],
                'short',
                self.peak_price
            )
            if current_price >= trailing_stop:
                self.close_position('BUY', current_data["low"].min())

    def close_position(self, side):
        """Close the active position and log trade details."""
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
                # Log the closed trade to CSV
                self.trade_logger.log_trade("SELL", 
                                            quantity=self.active_position['quantity'], 
                                            price=exit_price, 
                                            pnl=profit, 
                                            notes="Market sell order executed")
                self.active_position = None
                self.peak_price = None
        except Exception as e:
            logging.error(f"Error closing position: {e}")

    def execute_trade_cycle(self):
        """Main trading loop incorporating CSV logging and position status management."""
        logging.info("Starting trading cycle...")
        print(f"{Fore.CYAN}Starting trading bot...{Style.RESET_ALL}")
        print(f"Trading {Config.SYMBOL}USDT on {Config.TIMEFRAME} timeframe")
        print(f"Risk per trade: {Config.RISK_PER_TRADE*100}%")
        
        while True:
            try:
                self.check_position_status()
                data = self.get_historical_data()
                if data is None:
                    continue
                analyzed_data = self.strategy.analyze_market(data)
                signal = self.strategy.generate_signal(analyzed_data)
                self.print_market_status(analyzed_data, signal)
                
                # Manage active trade (if any)
                if self.active_position:
                    self.manage_active_trade(analyzed_data)
                else:
                    if self.position_status == 'ready_to_buy' and signal == 'long':
                        current_price = float(analyzed_data['close'].iloc[-1])
                        atr = float(analyzed_data['atr'].iloc[-1])
                        stop_distance = atr * 2
                        stop_loss = current_price - stop_distance
                        take_profit = current_price + (stop_distance * Config.MIN_RISK_REWARD)
                        quantity = self.calculate_position_size(current_price, stop_loss)
                        if quantity is None:
                            continue
                        order = self.place_order('BUY', quantity)
                        if order:
                            entry_price = float(order['fills'][0]['price'])
                            self.active_position = {
                                'type': 'long',
                                'entry_price': entry_price,
                                'quantity': quantity,
                                'entry_time': datetime.now(),
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            }
                            self.peak_price = entry_price
                            print(f"{Fore.GREEN}BUY order executed at {entry_price:.8f}{Style.RESET_ALL}")
                            print(f"Quantity: {quantity:.4f}")
                            print(f"Stop Loss: {stop_loss:.8f}")
                            print(f"Take Profit: {take_profit:.8f}")
                            # Log the BUY trade to CSV
                            self.trade_logger.log_trade("BUY", quantity=quantity, price=entry_price, notes="Market buy order executed")
                    elif self.position_status == 'ready_to_sell':
                        print(f"{Fore.YELLOW}Currently in SELL mode. Waiting for sell conditions to trigger exit...{Style.RESET_ALL}")
                
                # Countdown until next cycle (60 seconds)
                print(f"\n{Fore.YELLOW}Next update in:{Style.RESET_ALL}")
                for remaining in range(60, 0, -1):
                    print(f"\r{Fore.YELLOW}{remaining} seconds{Style.RESET_ALL}", end="")
                    time.sleep(1)
                print("\n")
                
            except Exception as e:
                print(f"{Fore.RED}Error in trade cycle: {str(e)}{Style.RESET_ALL}")
                logging.error(f"Error in trade cycle: {e}")
                time.sleep(60)

if __name__ == "__main__":
    API_KEY = os.getenv("API_KEY_")
    API_SECRET = os.getenv("API_SECRET_")
    trader = BinanceTradeExecutor(API_KEY, API_SECRET)
    trader.execute_trade_cycle()

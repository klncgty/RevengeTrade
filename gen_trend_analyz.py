import os
import time
import logging
from datetime import datetime
import pandas as pd
import talib
from binance.client import Client
from dotenv import load_dotenv
from config import Config
load_dotenv()

API_KEY = os.getenv("API_KEY_")
API_SECRET = os.getenv("API_SECRET_")
SYMBOL = Config.SYMBOL + "USDT"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def fetch_market_data(client, symbol, interval, limit=50):
    
    try:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} at interval {interval}: {e}")
        return None

def analyze_trend(df):
  
    if df is None or df.empty:
        return "Unknown"
    first_close = df['close'].iloc[0]
    last_close = df['close'].iloc[-1]
    trend = "Bullish" if last_close > first_close else "Bearish"
    return trend

def get_support_resistance(df):
    
    if df is None or df.empty:
        return (None, None)
    support = df['low'].min()
    resistance = df['high'].max()
    return support, resistance

def print_trend(trend_label, trend_direction, df):
    
    if df is None or df.empty:
        print(f"{trend_label}: No data available")
        return

    start_time = df['timestamp'].iloc[0].strftime("%Y-%m-%d %H:%M")
    end_time = df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M")
    support, resistance = get_support_resistance(df)
    print("========================================")
    print(f"{trend_label}: {trend_direction}")
    print(f"Time Window: {start_time} to {end_time}")
    if support is not None and resistance is not None:
        print(f"Support (Lowest Low): {support:.8f}")
        print(f"Resistance (Highest High): {resistance:.8f}")
    print("========================================\n")

def main():
    
    client = Client(API_KEY, API_SECRET, testnet=False, requests_params={'timeout': 30})
 
    timeframe_4h = "4h"   #  4 saatlik mum
    timeframe_1w = "1w"   # 1 haftalık mumlar
    
    df_4h = fetch_market_data(client, SYMBOL, timeframe_4h, limit=50)
    trend_4h = analyze_trend(df_4h)
    print_trend("4-Hour Trend", trend_4h, df_4h)
    
    df_1w = fetch_market_data(client, SYMBOL, timeframe_1w, limit=20)
    trend_1w = analyze_trend(df_1w)
    print_trend("Weekly Trend", trend_1w, df_1w)
    
if __name__ == "__main__":
    main()
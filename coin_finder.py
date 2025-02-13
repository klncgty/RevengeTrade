import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import numpy as np
from config import Config
import os   
import threading
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain_core.prompts import ChatPromptTemplate
import time
import talib
import logging

logging.getLogger("binance.client").setLevel(logging.WARNING)
logging.getLogger("langchain_groq").setLevel(logging.WARNING)

class CoinAnalyzer:
    def __init__(self, api_key, api_secret, coins):
        self.client = Client(api_key, api_secret)
        self.coins = coins
        
    
    def loading_animation(self):
        self.stop_loading = False
        while not self.stop_loading:
            for frame in "|/-\\":
                print(f"\rSeçilen coinlerin analizi yapılıyor {frame}", end="")
                time.sleep(0.1)
    def fetch_historical_data(self, symbol, interval='4h', lookback='30 days ago UTC'):
        klines = self.client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def analyze_trends(self):
        
        dakika = input("Kaç dakikalık alım satım yapacaksın:")
        print("\n")
        print("\033[94mconfig dosyasında timeframeleri değiştirebilirsin!! aksi halde işlemler 1 dakikalık yapılacak.\033[0m")
        print("\n")
        loading_thread = threading.Thread(target=self.loading_animation)
        loading_thread.start()
        
        volatility_data = {}
        analysis_results = {}
        for coin in self.coins:
            symbol = f"{coin}USDT"
            data = self.fetch_historical_data(symbol)
            
            # Fiyat ve hacim değişimlerini hesapla
            data['returns'] = data['close'].pct_change()
            data['volume_change'] = data['volume'].pct_change()

            # Hareketli Ortalamalar ile Trend Belirleme (TA-Lib)
            data['SMA_20'] = talib.SMA(data['close'], timeperiod=20)  # 20 günlük SMA
            data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)  # 20 günlük EMA

            uptrend = data['EMA_20'].iloc[-1] > data['SMA_20'].iloc[-1]  # EMA, SMA'dan büyükse yükseliş
            downtrend = data['EMA_20'].iloc[-1] < data['SMA_20'].iloc[-1]  # EMA, SMA'dan küçükse düşüş
            stable = data['returns'].std() < 0.01
            high_volume = data['volume_change'].mean() > 0.1

            # Volatilite Hesaplama (TA-Lib)
            volatility = talib.STDDEV(data['returns'], timeperiod=14).iloc[-1]

            # Bollinger Bandları ile Ekstra Trend Analizi
            upperband, middleband, lowerband = talib.BBANDS(data['close'], timeperiod=20)
            price_above_upper = data['close'].iloc[-1] > upperband.iloc[-1]  # Aşırı alım bölgesi
            price_below_lower = data['close'].iloc[-1] < lowerband.iloc[-1]  # Aşırı satım bölgesi

            volatility_data[symbol] = volatility
            analysis_results[symbol] = {
                "uptrend": uptrend,
                "downtrend": downtrend,
                "stable": stable,
                "high_volume": high_volume,
                "volatility": volatility,
                "price_above_upper_bb": price_above_upper,
                "price_below_lower_bb": price_below_lower,
            }
            
            
        # Determine the most volatile coin
        most_volatile_coin = max(volatility_data, key=volatility_data.get)
        analysis_results["most_volatile_coin"] = {
            "symbol": most_volatile_coin,
            "volatility": volatility_data[most_volatile_coin]
        }
        
        print("\n")
        print("\n")
        #print(f"\nThe most volatile coin is {most_volatile_coin} with a volatility of {volatility_data[most_volatile_coin]:.4f}")
        llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=0.7
            )
        
        
        prompt = ChatPromptTemplate.from_messages([
                ("system", Config.SYSTEM_PROMPT),
                
                ("human", "{text}"),
            ])
       

        chain = prompt | llm
        response = chain.invoke({"text": analysis_results,"dakika": dakika})
        # Stop loading animation
        self.stop_loading = True
        loading_thread.join()
               
        return response
    
 
        
if __name__ == "__main__":
    api_key = os.getenv("API_KEY_")
    api_secret = os.getenv("API_SECRET_")
    coins = ["SHIB", "BEAMX", "NOT", "SLP", "HOT","BTTC","PEPE","XEC","SPELL","COS","RVN"]
    analyzer = CoinAnalyzer(api_key, api_secret, coins)
    try:
        
        response = analyzer.analyze_trends()
        print("Cevap:", "\033[92m", response.content, "\033[0m")
    except Exception as e:
        print("An error occurred:", e)
import requests
import time
import numpy as np
import talib

### Likiditeleri kontrol eder. iyiyse buy sinyali üretir.
class TradingSignalGenerator:
    def __init__(self):
        pass

    # Veri Çekme Methodları
    def fetch_order_book(self, symbol, limit=100):
        """Emir defterini çeker."""
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
        response = requests.get(url)
        return response.json()

    def fetch_funding_rate(self, symbol):
        """Son fonlama oranını çeker."""
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        response = requests.get(url)
        return response.json()

    def fetch_open_interest(self, symbol):
        """Açık faiz verisini çeker."""
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        response = requests.get(url)
        return response.json()

    def fetch_klines(self, symbol, interval="15m", limit=5):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # HTTP hatalarını yakala
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Hatası: {str(e)}")
            return None

    # Analiz Methodları
    def analyze_market_makers(self, order_book):
        """Market Maker'ların hareketlerini analiz eder."""
        bids = order_book['bids']  # Alış emirleri
        asks = order_book['asks']  # Satış emirleri
        total_bid_volume = sum([float(bid[1]) for bid in bids])
        total_ask_volume = sum([float(ask[1]) for ask in asks])
        if total_bid_volume > total_ask_volume:
            return 1  # Alış emirleri baskın, buy sinyali
        else:
            return -1  # Satış emirleri baskın, sell sinyali

    def analyze_order_flow(self, order_book):
        """Order Flow'u analiz eder."""
        bids = order_book['bids']
        asks = order_book['asks']
        avg_bid_price = sum([float(bid[0]) for bid in bids]) / len(bids)
        avg_ask_price = sum([float(ask[0]) for ask in asks]) / len(asks)
        current_price = (float(bids[0][0]) + float(asks[0][0])) / 2
        if avg_bid_price > current_price and avg_ask_price > current_price:
            return 1  # Yükseliş eğilimi, buy sinyali
        elif avg_bid_price < current_price and avg_ask_price < current_price:
            return -1  # Düşüş eğilimi, sell sinyali
        else:
            return 0  # Nötr

    def analyze_funding_rate(self, funding_rate):
        """Funding Rate'i analiz eder."""
        rate = float(funding_rate[0]['fundingRate'])
        if rate > 0:
            return -1  # Pozitif oran, long pozisyonlar baskın, sell sinyali
        elif rate < 0:
            return 1  # Negatif oran, short pozisyonlar baskın, buy sinyali
        else:
            return 0  # Nötr

    def analyze_open_interest(self, symbol):
        # Tarihsel veri için önceki 2 periyodu çek
        try:
            current_data = self.fetch_open_interest(symbol)
            time.sleep(0.5)  # Rate limit için
            historical_data = self.fetch_open_interest(symbol)
            
            current_oi = float(current_data['openInterest'])
            previous_oi = float(historical_data['openInterest'])
            return 1 if current_oi > previous_oi else -1
        except Exception as e:
            print(f"Open Interest Hatası: {str(e)}")
            return 0

    def detect_stop_hunting(self, symbol):
        klines = self.fetch_klines(symbol, interval="5m", limit=50)  # Daha detaylı veri
        if not klines:
            return 0
            
        highs = [float(kline[2]) for kline in klines]
        lows = [float(kline[3]) for kline in klines]
        
        # Anomaly Detection: ATR'ye göre aykırı değerler
        atr = talib.ATR(
            high=np.array(highs),
            low=np.array(lows),
            close=np.array([float(kline[4]) for kline in klines]),
            timeperiod=14
        )[-1]
        
        last_low = lows[-1]
        if (lows[-2] - last_low) > (atr * 1.5):  # Anormal dip
            return 1  # Stop avı sonrası alım
        elif (highs[-2] - highs[-1]) > (atr * 1.5):  # Anormal tepe
            return -1
        return 0

    # Sinyal Üretimi
    def generate_signal_advanced(self, symbol):
        """Tüm analizleri birleştirerek nihai sinyali üretir."""
        order_book = self.fetch_order_book(symbol)
        funding_rate = self.fetch_funding_rate(symbol)
        
        mm_score = self.analyze_market_makers(order_book)
        of_score = self.analyze_order_flow(order_book)
        fr_score = self.analyze_funding_rate(funding_rate)
        oi_score = self.analyze_open_interest(symbol)
        sh_score = self.detect_stop_hunting(symbol)
        
        total_score = mm_score + of_score + fr_score + oi_score + sh_score
        
        if total_score > 0:
            return "buy"
        elif total_score < 0:
            return "sell"
        else:
            return "hold"

if __name__ == "__main__":
    generator = TradingSignalGenerator()
    signal = generator.generate_signal_advanced("1MBABYDOGEUSDT")
    print(f"Üretilen sinyal: {signal}")

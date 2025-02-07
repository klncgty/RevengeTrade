# #################### GELİŞMİŞ GÖSTERGELER ####################
import talib
from config import Config
class AdvancedIndicators:
    @staticmethod
    def calculate_all_indicators(data):
        # Temel Göstergeler
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        data['macd'], data['macd_signal'], _ = talib.MACD(data['close'])
        data['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Hacim Analizi
        data['volume_ma'] = talib.MA(data['volume'], timeperiod=20)
        data['volume_spike'] = data['volume'] > (data['volume_ma'] * Config.VOLUME_SPIKE_THRESHOLD)
        
        # Trend Analizi
        data['ema_50'] = talib.EMA(data['close'], timeperiod=50)
        data['ema_200'] = talib.EMA(data['close'], timeperiod=200)
        
        # Fiyat Desenleri
        data['sar'] = talib.SAR(data['high'], data['low'], acceleration=0.02, maximum=0.2)
        
        # Fibonacci Seviyeleri
        recent_high = data['high'].rolling(50).max().iloc[-1]
        recent_low = data['low'].rolling(50).min().iloc[-1]
        data['fib_38'] = recent_high - (recent_high - recent_low) * 0.382
        data['fib_62'] = recent_high - (recent_high - recent_low) * 0.618
        
        # Volatilite
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=Config.ATR_PERIOD)
        
        return data
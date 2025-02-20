# #################### GELİŞMİŞ GÖSTERGELER ####################
import talib
from config import Config
class AdvancedIndicators:
    @staticmethod
    def calculate_all_indicators(data):
        # Temel Göstergeler
        data['rsi'] = talib.RSI(data['close'], timeperiod=Config.TIME_PERIOD_RSI)
        data['macd'], data['macd_signal'], _ = talib.MACD(data['close'])
        data['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=Config.TIME_PERIOD_ADX)
        
        # Hacim Analizi
        data['volume_ma'] = talib.MA(data['volume'], timeperiod=20)
        data['volume_spike'] = data['volume'] > (data['volume_ma'] * Config.VOLUME_SPIKE_THRESHOLD)
        
        # Trend Analizi
        data['ema_50'] = talib.EMA(data['close'], timeperiod=50)
        data['ema_200'] = talib.EMA(data['close'], timeperiod=200)
        
        # Fiyat Desenleri
        data['sar'] = talib.SAR(data['high'], data['low'], acceleration=0.02, maximum=0.2)
        
        # Fibonacci Seviyeleri
        recent_high = data['high'].rolling(200).max().iloc[-1]
        recent_low = data['low'].rolling(200).min().iloc[-1]
        data['fib_38'] = recent_high - (recent_high - recent_low) * 0.382
        data['fib_62'] = recent_high - (recent_high - recent_low) * 0.618
        
        # Volatilite
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=Config.atr_period_supertrend)
        
        
        #supertrend 
        high = data['high']
        low = data['low']
        close = data['close']
        
        # ATR hesapla
        atr = talib.ATR(high, low, close, timeperiod=Config.atr_period_supertrend)
        hl2 = (high + low) / 2
        
        # Temel üst ve alt bantlar
        basic_upperband = hl2 + (Config.multiplier_supertrend * atr)
        basic_lowerband = hl2 - (Config.multiplier_supertrend * atr)

        # Temel bantlar
        basic_upperband = hl2 + (Config.multiplier_supertrend * atr)
        basic_lowerband = hl2 - (Config.multiplier_supertrend * atr)

        final_upperband = [0] * len(data)
        final_lowerband = [0] * len(data)
        supertrend = [True] * len(data)
        current_trend = [True] * len(data)
        # İlk değerler
        final_upperband[0] = basic_upperband.iloc[0]
        final_lowerband[0] = basic_lowerband.iloc[0]
        current_trend = [True] * len(data)
        trend_messages = [""] * len(data)
        last_trend_change = None
        for i in range(1, len(data)):
            current_price = close.iloc[i]
            previous_price = close.iloc[i-1]
            price_change = abs(current_price - previous_price) / previous_price * 100

            if current_trend[i-1]:  # Uptrend
                final_upperband[i] = min(basic_upperband.iloc[i], final_upperband[i-1])
                final_lowerband[i] = max(basic_lowerband.iloc[i], final_lowerband[i-1])
                
                if current_price < final_lowerband[i]:
                    current_trend[i] = False
                    last_trend_change = f"TREND CHANGE: Uptrend → Downtrend at price: {current_price:.2f}, Change: %{price_change:.2f}"
                else:
                    current_trend[i] = True
            else:  # Downtrend
                final_upperband[i] = max(basic_upperband.iloc[i], final_upperband[i-1])
                final_lowerband[i] = min(basic_lowerband.iloc[i], final_lowerband[i-1])
                
                if current_price > final_upperband[i]:
                    current_trend[i] = True
                    last_trend_change = f"TREND CHANGE: Downtrend → Uptrend at price: {current_price:.2f}, Change: %{price_change:.2f}"
                else:
                    current_trend[i] = False

            
            supertrend[i] = current_trend[i]
             # Only store the trend change message
            trend_messages[i] = last_trend_change if current_trend[i] != current_trend[i-1] else ""

        data['supertrend'] = supertrend
        data['final_upperband'] = final_upperband
        data['final_lowerband'] = final_lowerband
        data['trend_messages'] = trend_messages
        data['linear_reg_forecast'] = talib.LINEARREG(data['close'], timeperiod=14)
        
        
        return data
        
       
    
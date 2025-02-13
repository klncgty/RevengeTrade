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
        recent_high = data['high'].rolling(50).max().iloc[-1]
        recent_low = data['low'].rolling(50).min().iloc[-1]
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

        final_upperband = [0] * len(data)
        final_lowerband = [0] * len(data)
        supertrend = [True] * len(data)  # True = yükseliş trendi, False = düşüş trendi

        final_upperband[0] = basic_upperband.iloc[0]
        final_lowerband[0] = basic_lowerband.iloc[0]
        supertrend[0] = True  # İlk mumda varsayılan olarak yükseliş trendi

        for i in range(1, len(data)):
            # Üst bantın güncellenmesi
            if basic_upperband.iloc[i] < final_upperband[i-1] or close.iloc[i-1] > final_upperband[i-1]:
                final_upperband[i] = basic_upperband.iloc[i]
            else:
                final_upperband[i] = final_upperband[i-1]

            # Alt bantın güncellenmesi
            if basic_lowerband.iloc[i] > final_lowerband[i-1] or close.iloc[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = basic_lowerband.iloc[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]

            # SuperTrend sinyalinin belirlenmesi
            if close.iloc[i] <= final_upperband[i]:
                supertrend[i] = False  # Düşüş trendi sinyali
            else:
                supertrend[i] = True   # Yükseliş trendi sinyali

        data['supertrend'] = supertrend
        data['final_upperband'] = final_upperband
        data['final_lowerband'] = final_lowerband
        data['linear_reg_forecast'] = talib.LINEARREG(data['close'], timeperiod=14)
        
        return data
    
    
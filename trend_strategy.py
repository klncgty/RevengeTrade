# #################### TREND STRATEJİSİ ####################
from advanced_indicators import AdvancedIndicators
from dynamic_risk_manager import DynamicRiskManager
from config import Config

class TrendStrategy:
    def __init__(self):
        self.indicators = AdvancedIndicators()
        self.risk_manager = DynamicRiskManager()
        self.last_rsi_low = float('inf')
        self.last_price_low = float('inf')
        
    def analyze_market(self, data):
        data = self.indicators.calculate_all_indicators(data)
        return data

    def detect_divergence(self, data):
        """RSI ve fiyat arasındaki uyumsuzluğu tespit eder"""
        current_rsi = data['rsi'].iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Pozitif divergence (Fiyat düşerken RSI yükseliyor)
        if current_price < self.last_price_low and current_rsi > self.last_rsi_low:
            divergence = True
        else:
            divergence = False
            
        # Değerleri güncelle
        if current_rsi < self.last_rsi_low:
            self.last_rsi_low = current_rsi
        if current_price < self.last_price_low:
            self.last_price_low = current_price
            
        return divergence
    
    def generate_signal(self, data):
        current_close = data['close'].iloc[-1]
        volume_spike = data['volume_spike'].iloc[-1]
        # Debug 
        print("\nDebug - Buy Conditions Check:")
        rsi_check = data['rsi'].iloc[-1] < 50
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        
        macd_check = data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]
        ema_check = (current_close > data['ema_50'].iloc[-1] or 
                    current_close > data['ema_200'].iloc[-1])
        
        macd_momentum = macd > macd_signal and abs(macd - macd_signal) > abs(data['macd'].iloc[-2] - data['macd_signal'].iloc[-2])
        
        # EMA koşulları (daha yumuşak)
        ema_conditions = (
            # Fiyat EMA50'ye yaklaşıyor
            abs(current_close - data['ema_50'].iloc[-1]) / data['ema_50'].iloc[-1] < 0.01 or
            # Fiyat EMA200'e yaklaşıyor
            abs(current_close - data['ema_200'].iloc[-1]) / data['ema_200'].iloc[-1] < 0.015 or
            # EMA50 yükseliş trendinde
            data['ema_50'].iloc[-1] > data['ema_50'].iloc[-5]
        )
        divergence = self.detect_divergence(data)
        
        
        print(f"RSI Check ({data['rsi'].iloc[-1]:.2f} < 50): {rsi_check}")
        print(f"MACD Check: {macd_check}")
        print(f"EMA Check: {ema_check}")
        print(f"Volume Spike: {volume_spike}")
        # Çoklu Zaman Dilimi Trend Uyumu
        trend_alignment = self.check_trend_alignment(data)
        
        # Ana Trend Belirleme
        primary_trend = 'bullish' if data['ema_50'].iloc[-1] > data['ema_200'].iloc[-1] else 'bearish'
        fib_support_38 = current_close >= data['fib_38'].iloc[-1]  # 38.2% desteği
        fib_support_62 = current_close >= data['fib_62'].iloc[-1]  # 61.8% desteği
    
        # Alım Koşulları
        #buy_conditions = (
            #data['rsi'].iloc[-1] < 45 and
            #data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] and
            #current_close > data['ema_50'].iloc[-1] and
            #volume_spike and
            #trend_alignment == 'bullish' and
            #primary_trend == 'bullish'
            
            # DAHA AGRESİF ALIM KOŞULLARI
        """buy_conditions = (
            data['rsi'].iloc[-1] < 50 and                                
            data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] and    # MACD sinyali
            (current_close > data['ema_50'].iloc[-1] or                  # EMA50 VEYA EMA200 üzerinde olması yeterli
            current_close > data['ema_200'].iloc[-1]) and
            volume_spike    )
            #and  (fib_support_38 or fib_support_62)  # Fibonacci desteği ekledik"""
        
        # Agresif alım koşulları
        buy_conditions = (
            (data['rsi'].iloc[-1] < 50 or divergence) and  # RSI aşırı satım VEYA pozitif divergence
            (macd_momentum or volume_spike) and  # Momentum VEYA hacim artışı
            ema_conditions  # Yumuşatılmış EMA koşulları
        )
        print(f"Final Buy Decision: {buy_conditions}")
        # Satım Koşulları
        sell_conditions = (
            data['rsi'].iloc[-1] > 60 and
            data['macd'].iloc[-1] < data['macd_signal'].iloc[-1] and
            current_close < data['ema_50'].iloc[-1] and
            volume_spike and
            trend_alignment == 'bearish' and
            primary_trend == 'bearish'
        )
        
        if buy_conditions:
            return 'long'
        elif sell_conditions:
            return 'short'
        return 'hold'
    
    def check_trend_alignment(self, data):
        # 3 Farklı Zaman Dilimi Trend Kontrolü
        daily_trend = 'bullish' if data['close'].iloc[-1] > data['ema_200'].iloc[-1] else 'bearish'
        hourly_trend = 'bullish' if data['adx'].iloc[-1] > 25 and data['close'].iloc[-1] > data['ema_50'].iloc[-1] else 'bearish'
        momentum_trend = 'bullish' if data['sar'].iloc[-1] < data['close'].iloc[-1] else 'bearish'
        
        if daily_trend == hourly_trend == momentum_trend:
            return daily_trend
        return 'neutral'
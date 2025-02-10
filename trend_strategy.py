# #################### TREND STRATEJİSİ ####################
from advanced_indicators import AdvancedIndicators
from dynamic_risk_manager import DynamicRiskManager
from trend_channel_analyzer import TrendChannelAnalyzer
from typing import Dict, Optional, Tuple
from config import Config
import pandas as pd


class TrendStrategy:
    def __init__(self):
        self.indicators = AdvancedIndicators()
        self.risk_manager = DynamicRiskManager()
        self.channel_analyzer = TrendChannelAnalyzer()

        self.last_rsi_low = float('inf')
        self.last_price_low = float('inf')
        
    def calculate_entry_price(self, data: pd.DataFrame, upper_line: pd.Series, 
                            lower_line: pd.Series, position_type: str) -> float:
        """
        Alım veya satım için limit fiyatını hesaplar
        """
        current_close = data['close'].iloc[-1]
        current_upper = upper_line.iloc[-1]
        current_lower = lower_line.iloc[-1]
        channel_width = current_upper - current_lower
        
        if position_type == 'buy':
            # Alt kanala yakınsa
            if current_close < (current_lower + channel_width * 0.2):
                # Limit order fiyatı: Alt kanal + kanal genişliğinin %5'i
                entry_price = min(
                    current_close,
                    current_lower + channel_width * 0.05
                )
            else:
                entry_price = current_close
                
        else:  # sell
            # Üst kanala yakınsa
            if current_close > (current_upper - channel_width * 0.2):
                # Limit order fiyatı: Üst kanal - kanal genişliğinin %5'i
                entry_price = max(
                    current_close,
                    current_upper - channel_width * 0.05
                )
            else:
                entry_price = current_close
                
        return entry_price
    def calculate_target_and_stop(self, data: pd.DataFrame, entry_price: float,
                                upper_line: pd.Series, lower_line: pd.Series,
                                position_type: str) -> Tuple[float, float]:
        """
        Hedef fiyat ve stop-loss seviyelerini hesaplar
        """
        channel_width = upper_line.iloc[-1] - lower_line.iloc[-1]
        
        if position_type == 'buy':
            target_price = upper_line.iloc[-1]
            stop_loss = entry_price - (channel_width * 0.3)
        else:  # sell
            target_price = lower_line.iloc[-1]
            stop_loss = entry_price + (channel_width * 0.3)
            
        return target_price, stop_loss      
    def analyze_market(self, data):
        data = self.indicators.calculate_all_indicators(data)
        upper_line, lower_line = self.channel_analyzer.calculate_channel_lines(data)
        data['upper_channel'] = upper_line
        data['lower_channel'] = lower_line
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
    def check_channel_conditions(self, data, upper_line, lower_line):
        """Kanal koşullarını kontrol eder"""
        current_close = data['close'].iloc[-1]
        current_upper = upper_line.iloc[-1]
        current_lower = lower_line.iloc[-1]
        
        # Fiyatın kanaldaki pozisyonunu hesapla
        channel_position = (current_close - current_lower) / (current_upper - current_lower)
        
        # Kanal içi alım-satım bölgeleri
        near_lower = channel_position < 0.2  # Alt %20'lik dilim
        near_upper = channel_position > 0.8  # Üst %20'lik dilim
        return near_lower, near_upper, channel_position

    def generate_signal(self, data):
        current_close = data['close'].iloc[-1]
        volume_spike = data['volume_spike'].iloc[-1]
        
        # Kanal analizi
        upper_line, lower_line = self.channel_analyzer.calculate_channel_lines(data)
        near_lower, near_upper, channel_position = self.check_channel_conditions(data, upper_line, lower_line)
        
        # Debug çıktıları
        print("\nDebug - Buy Conditions Check:")
        rsi_check = data['rsi'].iloc[-1] < 55
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        
        macd_check = macd > macd_signal
        ema_check = (current_close > data['ema_50'].iloc[-1] or 
                    current_close > data['ema_200'].iloc[-1])
        
        macd_momentum = (macd > macd_signal and 
                        abs(macd - macd_signal) > abs(data['macd'].iloc[-2] - data['macd_signal'].iloc[-2]))
        
        # EMA koşulları
        ema_conditions = (
            abs(current_close - data['ema_50'].iloc[-1]) / data['ema_50'].iloc[-1] < 0.01 or
            abs(current_close - data['ema_200'].iloc[-1]) / data['ema_200'].iloc[-1] < 0.015 or
            data['ema_50'].iloc[-1] > data['ema_50'].iloc[-5]
        )
        
        divergence = self.detect_divergence(data)
        trend_alignment = self.check_trend_alignment(data)
        primary_trend = 'bullish' if data['ema_50'].iloc[-1] > data['ema_200'].iloc[-1] else 'bearish'
        
        print(f"RSI Check ({data['rsi'].iloc[-1]:.2f} < 55): {rsi_check}")
        print(f"MACD Check: {macd_check}")
        print(f"EMA Check: {ema_check}")
        print(f"Volume Spike: {volume_spike}")
        print(f"Channel Position: {channel_position:.2f}")
        print(f"Near Lower Channel: {near_lower}")
        print(f"Near Upper Channel: {near_upper}")
        
        # Geliştirilmiş alım koşulları
        buy_conditions = (
            ((data['rsi'].iloc[-1] < Config.RSI_BUY or divergence) and
            (macd_momentum or volume_spike) and
            ema_conditions) or
            (near_lower and macd_momentum)  # Kanal tabanında momentum varsa
        )
        
        # Geliştirilmiş satım koşulları
        sell_conditions = (
            (data['rsi'].iloc[-1] > 60 and
            data['macd'].iloc[-1] < data['macd_signal'].iloc[-1] and
            current_close < data['ema_50'].iloc[-1] and
            volume_spike and
            trend_alignment == 'bearish' and
            primary_trend == 'bearish') or
            (near_upper and macd < macd_signal)  # Kanal tavanında momentum düşüyorsa
        )
        
        print(f"Final Buy Decision: {buy_conditions}")
        
        if buy_conditions:
            # Alım fiyatını hesapla
            entry_price = self.calculate_entry_price(data, upper_line, lower_line)
            target_price = upper_line.iloc[-1]

            # Risk yönetimi
            channel_width = upper_line.iloc[-1] - lower_line.iloc[-1]
            stop_loss = entry_price - (channel_width * 0.3)  # Kanal genişliğinin %30'u kadar stop
            
            # Risk/Ödül oranını kontrol et
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            if reward / risk < Config.MIN_RISK_REWARD:
                return 'hold'
                
            print(f"Buy Signal:")
            print(f"Entry Price: {entry_price:.2f}")
            print(f"Target: {target_price:.2f}")
            print(f"Stop Loss: {stop_loss:.2f}")
            print(f"Risk/Reward: {reward/risk:.2f}")
            
            return {
                'signal': 'long',
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss
            }
        elif sell_conditions:
            target_price, stop_loss = self.channel_analyzer.calculate_target_prices(data, 'sell')
            print(f"Sell Signal - Target: {target_price:.2f}, Stop Loss: {stop_loss:.2f}")
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

# #################### TREND STRATEJİSİ ####################
from advanced_indicators import AdvancedIndicators
from dynamic_risk_manager import DynamicRiskManager
from trend_channel_analyzer import TrendChannelAnalyzer
from ema_reject_strategy import EMARejectStrategy
from typing import Dict, Optional, Tuple
from config import Config
import pandas as pd
from trade_database import TradePositionManager
from dataclasses import dataclass
import colorama
from colorama import Fore, Style
import logging

colorama.init()


@dataclass
class SafetyCheck:
    is_safe: bool
    message: str
    current_price: float
    entry_price: Optional[float] = None

@dataclass
class PositionInfo:
    total_quantity: float
    entry_price: float
    remaining_quantity: float
    partial_sells: list

class RiskCheck:
    def __init__(self, db: Optional[TradePositionManager] = None):
        self.db = db
        
    def check_sell_safety(self, symbol: str, current_price: float) -> SafetyCheck:
        """
        Satış işleminin güvenli olup olmadığını kontrol eder
        """
        active_position = self.db.get_active_position(symbol)
        
        if not active_position:
            return SafetyCheck(True, "No active position", current_price)
            
        entry_price = active_position['entry_price']
        
        if current_price < entry_price:
            return SafetyCheck(
                is_safe=False,
                message=f"Current price ({current_price:.8f}) is below entry price ({entry_price:.8f})",
                current_price=current_price,
                entry_price=entry_price
            )
            
        return SafetyCheck(
            is_safe=True,
            message="Price is above entry price",
            current_price=current_price,
            entry_price=entry_price
        )

class TrendStrategy:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.indicators = AdvancedIndicators()
        self.risk_manager = DynamicRiskManager()
        self.channel_analyzer = TrendChannelAnalyzer()
        self.db = TradePositionManager()
        self.risk_check = RiskCheck(self.db)
        self.last_rsi_low = float('inf')
        self.last_price_low = float('inf')
        self.active_position = None
        self.peak_price = None
        self.pending_sell_order = None
        self.sell_order_time = None
        self.max_daily_loss = -2.0  # Maximum daily loss percentage
        self.max_position_size = 1000  # Maximum position size in USDT
        self.position_info = None
        self.daily_high = float('-inf')
        self.PARTIAL_SELL_PERCENTAGE = 40  # Sell 40% of remaining position
        self.ema_reject = EMARejectStrategy()

    def update_daily_high(self, current_price: float):
        """Updates the daily high price"""
        self.daily_high = max(self.daily_high, current_price)

    def reset_daily_high(self):
        """Resets daily high (should be called at the start of each day)"""
        self.daily_high = float('-inf')

    def calculate_sell_quantity(self, total_quantity: float) -> float:
        """Calculates the quantity to sell based on partial sell percentage"""
        return total_quantity * (self.PARTIAL_SELL_PERCENTAGE / 100)

    def check_partial_sell_conditions(self, data: pd.DataFrame) -> bool:
        """
        Checks conditions for partial sell:
        1. All regular sell conditions are met
        2. RSI is below 50
        3. Current price is above entry price
        4. Current price is the daily high
        """
        if not self.position_info:
            return False

        current_price = float(data['close'].iloc[-1])
        rsi = data['rsi'].iloc[-1]
        
        # Regular sell conditions
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        volume_spike = data['volume_spike'].iloc[-1]
        trend_alignment = self.check_trend_alignment(data)
        
        technical_conditions = (
            rsi < 50 and  # New condition: RSI below 50
            macd < macd_signal and
            current_price < data['ema_50'].iloc[-1] and
            trend_alignment == 'bearish' and
            volume_spike
        )
        

        # Price conditions
        price_conditions = (
            current_price > self.position_info.entry_price and
            abs(current_price - self.daily_high) < 0.0000001  # Using small epsilon for float comparison
        )

        return technical_conditions and price_conditions
    
    def perform_safety_check(self, current_price: float, position_size: float, 
                           daily_pnl: float, entry_price: Optional[float] = None) -> SafetyCheck:
        """
        Performs comprehensive safety checks before executing trades
        """
        # Check daily loss limit
        if daily_pnl < self.max_daily_loss:
            return SafetyCheck(
                is_safe=False,
                message="Daily loss limit reached",
                current_price=current_price,
                entry_price=entry_price
            )

        # Check position size
        if position_size > self.max_position_size:
            return SafetyCheck(
                is_safe=False,
                message="Position size exceeds maximum limit",
                current_price=current_price,
                entry_price=entry_price
            )

        # Check for extreme volatility
        if entry_price and abs((current_price - entry_price) / entry_price * 100) > 5:
            return SafetyCheck(
                is_safe=False,
                message="Price movement too volatile",
                current_price=current_price,
                entry_price=entry_price
            )

        # All checks passed
        return SafetyCheck(
            is_safe=True,
            message="All safety checks passed",
            current_price=current_price,
            entry_price=entry_price
        )

    def check_market_conditions(self, data: pd.DataFrame) -> SafetyCheck:
        """
        Checks market conditions for safe trading
        """
        current_price = float(data['close'].iloc[-1])
        
        # Check volume
        if data['volume'].iloc[-1] < data['volume'].rolling(20).mean().iloc[-1] * 0.5:
            return SafetyCheck(
                is_safe=False,
                message="Volume too low",
                current_price=current_price
            )

        # Check spread
        if 'high' in data and 'low' in data:
            spread = (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['low'].iloc[-1] * 100
            if spread > 1.5:  # If spread is more than 1.5%
                return SafetyCheck(
                    is_safe=False,
                    message="Spread too high",
                    current_price=current_price
                )

        return SafetyCheck(
            is_safe=True,
            message="Market conditions are suitable",
            current_price=current_price
        )
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
        near_lower = channel_position < 0.45  # Alt %20'lik dilim
        near_upper = channel_position > 0.8  # Üst %20'lik dilim
        return near_lower, near_upper, channel_position
    def check_order_status(self, symbol: str = Config.SYMBOL) -> None:
        """Bekleyen emirlerin durumunu kontrol eder"""
        try:
            if self.pending_sell_order:
                order = self.client.get_order(
                    symbol=f"{symbol}USDT",
                    orderId=self.pending_sell_order['orderId']
                )
                
                if order['status'] == 'FILLED':
                    print(f"{Fore.GREEN}Sell order filled at {order['price']}{Style.RESET_ALL}")
                    self.pending_sell_order = None
                    self.sell_order_time = None
                    self.active_position = None
                    
        except Exception as e:
            logging.error(f"Error checking order status: {e}")
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
       
        try:
            # Dinamik hedef hesaplaması
            long_target, short_target = self.risk_manager.calculate_dynamic_target(data)
            
            # Eğer hedef hesaplanamadıysa
            if long_target is None or short_target is None:
                self.logger.error("Hedef hesaplama başarısız")
                return "hold"
            
            current_price = float(data['close'].iloc[-1])
            self.update_daily_high(current_price)
            
            # EMA rejection kontrolü: Eğer aktif pozisyon varsa ve EMA rejection tetikleniyorsa, satış sinyali ver.
            if self.active_position and self.ema_reject.analyze_ema_rejections(data):
                print(f"{Fore.RED}EMA rejection detected - Generating sell signal{Style.RESET_ALL}")
                return 'short'
            
            # Emir süresi dolmuşsa EMA rejection sayaçlarını sıfırla.
            if self.ema_reject.check_sell_order_expiry():
                self.ema_reject.reset_rejection_count()
            
            # Veritabanından pozisyon bilgisini güncelle (eğer yoksa)
            if not self.position_info:
                active_position = self.db.get_active_position(Config.SYMBOL)
                if active_position:
                    self.position_info = PositionInfo(
                        total_quantity=active_position['quantity'],
                        entry_price=active_position['entry_price'],
                        remaining_quantity=active_position['quantity'],
                        partial_sells=[]
                    )
            
            # PARTIAL SELL: Eğer pozisyon bilgisi varsa ve kısmi satış koşulları sağlanıyorsa, satış sinyali ver.
            if self.position_info and self.position_info.remaining_quantity > 0:
                if self.check_partial_sell_conditions(data):
                    sell_quantity = self.calculate_sell_quantity(self.position_info.remaining_quantity)
                    print(f"\n{Fore.YELLOW}=== Partial Sell Signal Generated ==={Style.RESET_ALL}")
                    print(f"Current Price: {current_price:.8f}")
                    print(f"Entry Price: {self.position_info.entry_price:.8f}")
                    print(f"Selling {self.PARTIAL_SELL_PERCENTAGE}% of remaining position")
                    print(f"Sell Quantity: {sell_quantity:.8f}")
                    return "short"
            
            # Güvenlik kontrolü: Eğer güvenli değilse hiçbir işlem yapılmadan hold sinyali ver.
            safety_check = self.risk_check.check_sell_safety(Config.SYMBOL, current_price)
            if not safety_check.is_safe:
                print(f"\n{Fore.YELLOW}Safety Check Warning: {safety_check.message}{Style.RESET_ALL}")
                return "hold"
            
            # Teknik göstergeler ve analizler
            rsi = data['rsi'].iloc[-1]
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            volume_spike = data['volume_spike'].iloc[-1]
            current_close = data['close'].iloc[-1]
            
            # Kanal analizi
            upper_line, lower_line = self.calculate_channel_lines(data)
            predicted_price = self.predict_next_candle_price(data)
            channel_width = upper_line.iloc[-1] - lower_line.iloc[-1]
            channel_position = (current_price - lower_line.iloc[-1]) / channel_width
            near_lower = channel_position < 0.2
            near_upper = channel_position > 0.8
            
            # Divergence kontrolü
            divergence = self.detect_divergence(data)
            
            # Fibonacci destekleri
            fib_support_38 = current_close >= data['fib_38'].iloc[-1]
            fib_support_62 = current_close >= data['fib_62'].iloc[-1]
            
            # Trend analizi
            trend_alignment = self.check_trend_alignment(data)
            primary_trend = 'bullish' if data['ema_50'].iloc[-1] > data['ema_200'].iloc[-1] else 'bearish'
            
            # SATIŞ KOŞULLARI (sadece aktif pozisyon varsa)
            if self.active_position:
                entry_price = self.active_position['entry_price']
                
                # Emergency Sell: Haftalık yüksek veya % belirli kar sağlanmışsa satış sinyali ver.
                if Config.EMERGENCY_SELL_ENABLED:
                    weekly_high = data['high'].rolling(window=168).max().iloc[-1]  # 168 = 7 gün * 24 saat
                    profit_percentage = ((current_price - entry_price) / entry_price) * 100
                    if current_price >= weekly_high or profit_percentage >= Config.EMERGENCY_SELL_PERCENTAGE:
                        print(f"\n{Fore.GREEN}=== Emergency Sell Signal ==={Style.RESET_ALL}")
                        print(f"Weekly High: {weekly_high:.8f}")
                        print(f"Profit: {profit_percentage:.2f}%")
                        return 'short'
                
                # Teknik Satış Koşulları
                rsi_sell_condition = rsi > Config.RSI_SELL
                macd_sell_condition = macd < macd_signal
                ema_sell_condition = current_price < data['ema_50'].iloc[-1]
                trend_sell_condition = trend_alignment == 'bearish'
                technical_sell = (rsi_sell_condition and macd_sell_condition and ema_sell_condition and trend_sell_condition and volume_spike and primary_trend == 'bearish')
                price_prediction_check = current_price >= predicted_price > entry_price
                
                if technical_sell and price_prediction_check:
                    print(f"\n{Fore.GREEN}=== Technical Sell Signal Generated ==={Style.RESET_ALL}")
                    return 'short'
            
            # ALIŞ KOŞULLARI
            supertrend_condition = data['supertrend'].iloc[-1]
            rsi_check = rsi < Config.RSI_BUY
            macd_momentum = (macd > macd_signal and 
                            abs(macd - macd_signal) > abs(data['macd'].iloc[-2] - data['macd_signal'].iloc[-2]))
            ema_conditions = (
                abs(current_price - data['ema_50'].iloc[-1]) / data['ema_50'].iloc[-1] < 0.01 or
                abs(current_price - data['ema_200'].iloc[-1]) / data['ema_200'].iloc[-1] < 0.015 or
                data['ema_50'].iloc[-1] > data['ema_50'].iloc[-5]
            )
            forecast_condition = current_price < data['linear_reg_forecast'].iloc[-1]
            buy_conditions = (
                (rsi_check or divergence) and      # RSI veya pozitif divergence
                (macd_momentum or volume_spike) and  # Momentum veya hacim artışı
                ema_conditions and                   # EMA yakınlığı
                near_lower and                       # Kanalın alt sınırına yakınlık
                (fib_support_38 or fib_support_62) and  # Fibonacci desteği
                current_price <= predicted_price and
                supertrend_condition      and forecast_condition           # Fiyat tahminine uyum
            )
            print(f"{Fore.CYAN}Buy_Conditions: {buy_conditions}{Style.RESET_ALL}")

            # Alış koşullarını detaylı yazdırma
            print(f"\n{Fore.CYAN}=== Buy Conditions ==={Style.RESET_ALL}")
            print(f"RSI < {Config.RSI_BUY}: {'✓' if rsi_check else '✗'} ({rsi:.2f})")
            print(f"MACD/Volume: {'✓' if (macd_momentum or volume_spike) else '✗'}")
            print(f"EMA Conditions: {'✓' if ema_conditions else '✗'}")
            print(f"Near Lower Channel: {'✓' if near_lower else '✗'}")
            print(f"Price Prediction: {'✓' if current_price <= predicted_price else '✗'}")
            print(f"SuperTrend (Uptrend): {'✓' if supertrend_condition else '✗'}")
            print(f"Forecast condition: {'✓' if forecast_condition else '✗'}")
            print(f"Fibonacci Support: {'✓' if fib_support_38 or fib_support_62 else '✗'}")
            
            
            # Eğer alış koşulları sağlanıyorsa, risk/ödül kontrolü yapılarak alış sinyali üretilir.
            if buy_conditions:
                logging.info("Buy conditions met, generating long signal.")
                target_price = upper_line.iloc[-1]
                stop_loss = current_price - (channel_width * 0.3)
                risk = current_price - stop_loss
                reward = target_price - current_price
                
                if risk > 0 and reward / risk >= Config.MIN_RISK_REWARD:
                    print(f"\n{Fore.GREEN}=== Buy Signal Generated ==={Style.RESET_ALL}")
                    print(f"Entry Price: {current_price:.8f}")
                    print(f"Target: {target_price:.8f}")
                    print(f"Stop Loss: {stop_loss:.8f}")
                    print(f"Risk/Reward: {reward/risk:.2f}")
                    return "long"
                else:
                    print(f"\n{Fore.YELLOW}Buy conditions met but R/R ratio insufficient{Style.RESET_ALL}")
                    return "hold"  # Risk/Ödül oranı yeterli değilse "hold" döndür

            # Buraya sadece buy_conditions SAĞLANMADIĞINDA girsin
            logging.info("Buy conditions not met, no long signal generated.")
            return "hold"

        except Exception as e:
            self.logger.error(f"Signal üretiminde hata: {str(e)}")
            return "hold"

    def update_position_after_partial_sell(self, sold_quantity: float, sell_price: float):
        """Updates position information after a partial sell"""
        if self.position_info:
            self.position_info.remaining_quantity -= sold_quantity
            self.position_info.partial_sells.append({
                'quantity': sold_quantity,
                'price': sell_price,
                'timestamp': pd.Timestamp.now()
            })
    
    def check_trend_alignment(self, data):
        # 3 Farklı Zaman Dilimi Trend Kontrolü
        daily_trend = 'bullish' if data['close'].iloc[-1] > data['ema_200'].iloc[-1] else 'bearish'
        hourly_trend = 'bullish' if data['adx'].iloc[-1] > 25 and data['close'].iloc[-1] > data['ema_50'].iloc[-1] else 'bearish'
        momentum_trend = 'bullish' if data['sar'].iloc[-1] < data['close'].iloc[-1] else 'bearish'
        
        if daily_trend == hourly_trend == momentum_trend:
            return daily_trend
        return 'neutral'
    def predict_next_candle_price(self, data: pd.DataFrame) -> float:
       
        """
        Gelişmiş kanal analizi ve momentum göstergeleri kullanarak
        bir sonraki mum fiyatını tahmin eder
        """
        try:
            current_price = data['close'].iloc[-1]
            upper_line, lower_line = self.calculate_channel_lines(data)
            
            # Momentum göstergeleri
            rsi = data['rsi'].iloc[-1]
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            
            # Kanal pozisyonu hesaplama
            channel_width = upper_line.iloc[-1] - lower_line.iloc[-1]
            channel_position = (current_price - lower_line.iloc[-1]) / channel_width
            
            # Fiyat tahmini için faktörler
            momentum_factor = 1.0
            
            # RSI bazlı momentum
            if rsi < 30:  # Aşırı satım
                momentum_factor += 0.005
            elif rsi > 70:  # Aşırı alım
                momentum_factor -= 0.005
                
            # MACD bazlı momentum
            if macd > macd_signal:
                momentum_factor += 0.003
            else:
                momentum_factor -= 0.003
                
            # Kanal pozisyonuna göre tahmin
            if channel_position > 0.7:  # Üst kanala yakın
                predicted_price = current_price * 0.995
            elif channel_position < 0.3:  # Alt kanala yakın
                predicted_price = current_price * 1.005
            else:
                predicted_price = current_price * momentum_factor
                    
            return predicted_price
                
        except Exception as e:
            logging.error(f"Fiyat tahmin hatası: {e}")
            return current_price
        
    def calculate_channel_lines(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates trend channel lines using high/low prices
        """
        try:
            # 20 periyotluk yüksek ve düşük fiyatları hesapla
            high_prices = data['high'].rolling(window=20).max()
            low_prices = data['low'].rolling(window=20).min()
            
            # Kanal genişliğini hesapla
            channel_width = high_prices - low_prices
            
            # Dinamik kanal sınırlarını belirle
            upper_line = high_prices + (channel_width * 0.1)
            lower_line = low_prices - (channel_width * 0.1)
            
            return upper_line, lower_line
            
        except Exception as e:
            self.logger.error(f"Channel calculation error: {e}")
            return pd.Series(), pd.Series()    
    def print_channel_status(self, data: pd.DataFrame) -> None:
        """
        Prints current channel status and price predictions
        """
        try:
            current_price = data['close'].iloc[-1]
            upper_line, lower_line = self.calculate_channel_lines(data)
            predicted_price = self.predict_next_candle_price(data)
            
            channel_width = upper_line.iloc[-1] - lower_line.iloc[-1]
            channel_position = (current_price - lower_line.iloc[-1]) / channel_width
            
            print("\n=== Channel Analysis ===")
            print(f"Current Price: {current_price:.8f}")
            print(f"Upper Channel: {upper_line.iloc[-1]:.8f}")
            print(f"Lower Channel: {lower_line.iloc[-1]:.8f}")
            print(f"Channel Width: {channel_width:.8f}")
            print(f"Channel Position: {channel_position:.2%}")
            print(f"Predicted Next Price: {predicted_price:.8f}")
            print(f"Expected Move: {((predicted_price/current_price)-1)*100:.2f}%")
            print("=====================")
            
        except Exception as e:
            print(f"Error printing channel status: {e}")
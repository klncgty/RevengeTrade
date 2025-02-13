import talib
import numpy as np
from config import Config
import logging
from typing import Optional, Tuple, Union

class DynamicRiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_dynamic_target(self, data):
        """
        Dinamik hedef fiyatları hesaplar ve hata kontrolü yapar
        """
        try:
            # Veri kontrolü
            if data is None or data.empty:
                raise ValueError("Geçersiz veri")
                
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("Eksik fiyat kolonları")

            # NaN değer kontrolü
            if data['high'].isnull().any() or data['low'].isnull().any() or data['close'].isnull().any():
                self.logger.warning("Veri setinde NaN değerler var - temizleniyor")
                data = data.fillna(method='ffill')

            # ATR hesaplama
            atr = talib.ATR(data['high'], 
                           data['low'], 
                           data['close'], 
                           timeperiod=Config.ATR_PERIOD)
            
            if atr.iloc[-1] <= 0:
                raise ValueError("Geçersiz ATR değeri")

            current_atr = atr.iloc[-1]
            close_price = data['close'].iloc[-1]

            # Hedef hesaplama
            long_target = close_price + (current_atr * 2)
            short_target = close_price - (current_atr * 1.5)

            return long_target, short_target

        except Exception as e:
            self.logger.error(f"Hedef hesaplama hatası: {str(e)}")
            return None, None

    def trailing_stop(self, 
                     current_price: float, 
                     entry_price: float, 
                     position_type: str,
                     peak_price: Optional[float] = None) -> Optional[float]:
       
        try:
            if not all(isinstance(x, (int, float)) for x in [current_price, entry_price]):
                raise ValueError("Invalid price values")

            if current_price <= 0 or entry_price <= 0:
                raise ValueError("Prices must be positive")

            if position_type not in ['long', 'short']:
                raise ValueError("Invalid position type")

            # Use current_price as peak_price if none provided
            peak_price = peak_price if peak_price is not None else current_price

            if position_type == 'long':
                new_stop = current_price * (1 - Config.TRAILING_STOP_PCT/100)
                return max(new_stop, peak_price * (1 - Config.TRAILING_STOP_PCT/100))
            else:
                new_stop = current_price * (1 + Config.TRAILING_STOP_PCT/100)
                return min(new_stop, peak_price * (1 + Config.TRAILING_STOP_PCT/100))

        except Exception as e:
            self.logger.error(f"Trailing stop calculation error: {str(e)}")
            return None

    def validate_risk_params(self):
        """
        Risk parametrelerinin geçerliliğini kontrol eder
        """
        try:
            if Config.ATR_PERIOD <= 0:
                raise ValueError("ATR periyodu pozitif olmalı")
            
            if Config.TRAILING_STOP_PCT <= 0 or Config.TRAILING_STOP_PCT >= 100:
                raise ValueError("Trailing stop yüzdesi 0-100 arasında olmalı")

            return True

        except Exception as e:
            self.logger.error(f"Risk parametre kontrolü hatası: {str(e)}")
            return False
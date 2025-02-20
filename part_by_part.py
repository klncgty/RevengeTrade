from dataclasses import dataclass
from typing import Dict, Optional
import logging
from datetime import datetime

# Log konfigürasyonu
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class TradePart:
    price_change: float
    allocation: float
    executed: bool = False
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None

    def execute(self, current_price: float):
        """TradePart için execute işlemini gerçekleştirir."""
        if self.executed:
            logging.debug("Bu TradePart zaten execute edilmiş.")
            return
        self.execution_price = current_price
        self.execution_time = datetime.now()
        self.executed = True
        logging.info(f"TradePart execute edildi: fiyat={current_price}, zaman={self.execution_time}")

    def reset(self):
        """TradePart durumunu sıfırlar."""
        self.executed = False
        self.execution_price = None
        self.execution_time = None

class PartByPartStrategy:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Alım seviyeleri
        self.buy_parts = [
            TradePart(price_change=0.4, allocation=0.25),
            TradePart(price_change=0.8, allocation=0.25),
            TradePart(price_change=1.2, allocation=0.50),
        ]
        # Satım seviyeleri
        self.sell_parts = [
            TradePart(price_change=0.7, allocation=0.40),
            TradePart(price_change=1.5, allocation=0.30),
            TradePart(price_change=2.5, allocation=0.30),
        ]
        self.reference_price = None
        self.last_execution_price = None
        self._is_initialized = False

    def initialize_reference_price(self, current_price: float):
        """Referans fiyatı sadece bir kez başlangıçta set eder."""
        if not self._is_initialized:
            self.reference_price = current_price
            self._is_initialized = True
            self.logger.info(f"Referans fiyat ilk kez set edildi: {current_price}")

    def calculate_price_change(self, current_price: float, reference_price: float) -> float:
        return ((current_price - reference_price) / reference_price) * 100

    def check_buy_conditions(self, current_price: float, four_hour_low=False) -> Dict:
        if self.reference_price is None:
            self.logger.warning("Referans fiyat yok; işlem yapılamaz. Referans fiyat set ediliyor.")
            self.reference_price = current_price
            return {'should_buy': False, 'allocation': 0, 'message': 'Referans fiyat yok, işlem yapılmadı.'}

        price_drop = -self.calculate_price_change(current_price, self.reference_price)
        self.logger.debug(f"Price drop: {price_drop:.2f}%")

        # 📌 Ana koşul: Yeterli fiyat düşüşü olup olmadığını kontrol et
        for part in self.buy_parts:
            if not part.executed and price_drop >= part.price_change:
                part.execute(current_price)
                self.reference_price = current_price
                self.last_execution_price = current_price
                return {
                    'should_buy': True,
                    'allocation': part.allocation,
                    'message': "💫 Buy Signal Generated!"
                }
        
        # 📌 Alternatif koşul: 4 saatlik en düşük fiyat bölgesinde mi?
        if four_hour_low:
            for part in self.buy_parts:
                if not part.executed:
                    part.execute(current_price)
                    self.reference_price = current_price
                    self.last_execution_price = current_price
                    return {
                        'should_buy': True,
                        'allocation': part.allocation,
                        'message': "💫 Buy Signal Generated (4H Low)!"
                    }

        return {'should_buy': False, 'allocation': 0, 'message': 'Koşullar sağlanmadı, işlem yapılmadı.'}


    def check_sell_conditions(self, current_price: float, entry_price: float) -> Dict:
        price_gain = self.calculate_price_change(current_price, entry_price)
        self.logger.debug(f"Price gain: {price_gain:.2f}%")
        for part in self.sell_parts:
            if not part.executed and price_gain >= part.price_change:
                part.execute(current_price)
                return {
                    'should_sell': True,
                    'allocation': part.allocation,
                    'message': "💫 Sell Signal Generated!"
                }
        return {'should_sell': False, 'allocation': 0, 'message': ''}

    def get_next_target_drop(self) -> float:
        """Bir sonraki alım hedefi için düşüş yüzdesini hesaplar."""
        for part in self.buy_parts:
            if not part.executed:
                return part.price_change / 100
        return 0.0

    def reset(self):
        for part in self.buy_parts + self.sell_parts:
            part.reset()
        # İsteğe bağlı: referans fiyat da resetlenebilir.
        # self.reference_price = None

# Eğer bu modülü doğrudan çalıştırırsanız demo amaçlı örnek
if __name__ == "__main__":
    strategy = PartByPartStrategy()
    current_price = 100.0
    strategy.initialize_reference_price(current_price)

    # Fiyat hareketlerini simüle edin.
    test_prices = [98, 96, 95, 93, 97, 102, 105]
    for price in test_prices:
        buy_signal = strategy.check_buy_conditions(price)
        if buy_signal['should_buy']:
            print(f"Buy signal at price {price}: {buy_signal}")
        sell_signal = strategy.check_sell_conditions(price, current_price)
        if sell_signal['should_sell']:
            print(f"Sell signal at price {price}: {sell_signal}")
            
    strategy.reset()
    print("Strateji resetlendi.")
    
    test_prices = [150,51 , 95, 45, 97, 102, 105]
    for price in test_prices:
        buy_signal = strategy.check_buy_conditions(price)
        if buy_signal['should_buy']:
            print(f"Buy signal at price {price}: {buy_signal}")
        sell_signal = strategy.check_sell_conditions(price, current_price)
        if sell_signal['should_sell']:
            print(f"Sell signal at price {price}: {sell_signal}")
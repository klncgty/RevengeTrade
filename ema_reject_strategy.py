from datetime import datetime, timedelta
import pandas as pd
import logging

class EMARejectStrategy:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rejection_count = 0
        self.last_rejection_price = None
        self.rejection_timestamps = []
        self.max_rejections = 3  #  3 defa reddedildikten sonra satış emri koy
        self.pending_sell_order = None  # Açık satış emri
        self.sell_order_expiry = None  # Satış emri geçerlilik süresi

    def reset_rejection_count(self):
        """Rejection sayaçlarını sıfırla"""
        self.rejection_count = 0
        self.last_rejection_price = None
        self.rejection_timestamps = []
        self.pending_sell_order = None
        self.sell_order_expiry = None

    def analyze_ema_rejections(self, data: pd.DataFrame):
        """
        EMA 50 seviyesinde fiyatın kaç defa reddedildiğini analiz eder
        """
        try:
            current_price = data['close'].iloc[-1]
            ema_50 = data['ema_50'].iloc[-1]
            timestamp = data.index[-1]

            # Fiyat EMA50'nin üstüne çıkıyor ama tutunamıyorsa sayaç artır
            if current_price > ema_50:
                if data['close'].iloc[-2] <= data['ema_50'].iloc[-2]:  # Önceki fiyat EMA'nın altındaysa
                    self.rejection_count += 1
                    self.last_rejection_price = current_price
                    self.rejection_timestamps.append(timestamp)
                    print(f"Rejection #{self.rejection_count} - Price: {self.last_rejection_price}")

            if self.rejection_count >= self.max_rejections and not self.pending_sell_order:
                self.pending_sell_order = self.last_rejection_price 
                self.sell_order_expiry = datetime.now() + timedelta(minutes=2)
                print(f"SATIŞ EMRİ: {self.pending_sell_order} fiyatından 2 dakika geçerli!")
                return True  # Satış sinyali üret
            return False
        except Exception as e:
            self.logger.error(f"EMA rejection analizi sırasında hata: {e}")

    def check_sell_order_expiry(self):
        """Satış emrinin süresi dolduysa iptal et"""
        if self.pending_sell_order and datetime.now() > self.sell_order_expiry:
            print(f"Satış emri iptal edildi: {self.pending_sell_order}")
            self.reset_rejection_count()
            return True
        return False

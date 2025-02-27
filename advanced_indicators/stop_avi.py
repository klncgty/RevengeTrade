import requests
import time

class StopAviDedektoru:
    def __init__(self, symbol="1MBABYDOGEUSDT"):
        self.symbol = symbol

    
    def get_order_book(self, limit=100):
        url = f"https://api.binance.com/api/v3/depth?symbol={self.symbol}&limit={limit}"
        response = requests.get(url).json()
        return response

   
    def get_open_interest(self):
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={self.symbol}"
        response = requests.get(url).json()
        return float(response['openInterest'])


    def get_funding_rate(self):
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={self.symbol}"
        response = requests.get(url).json()
        return float(response['lastFundingRate'])


    def stop_avi_kontrol(self):
        try:
            print("\n📊 Stop Avı Dedektörü Çalışıyor...")

            # **1. Order Book Dengesizliği Analizi**
            order_book = self.get_order_book()
            bids = sum(float(b[1]) for b in order_book["bids"])  # Alış likiditesi
            asks = sum(float(a[1]) for a in order_book["asks"])  # Satış likiditesi
            imbalance = abs(bids - asks) / (bids + asks) * 100  # Yüzdesel fark

            if imbalance > 10:
                print(f"🚨 STOP AVI RİSKİ! Order Book dengesiz: %{imbalance:.2f}")
                return True  # Stop Avı riski var, işlem yapma

            # **2. Open Interest (OI) Ani Düşüşü Kontrolü**
            open_interest_before = self.get_open_interest()
            time.sleep(5)
            open_interest_after = self.get_open_interest()
            oi_change = (open_interest_after - open_interest_before) / open_interest_before * 100

            if oi_change < -2:
                print(f"🚨 STOP AVI RİSKİ! Open Interest ani düştü: %{oi_change:.2f}")
                return True

            # **3. Funding Rate Manipülasyonu Kontrolü**
            funding_rate = self.get_funding_rate()
            if abs(funding_rate) > 0.02:
                print(f"🚨 STOP AVI RİSKİ! Funding Rate aşırı yüksek: %{funding_rate:.4f}")
                return True

            print("✅ Stop Avı Riski Yok. İşlem güvenli.")
            return False

        except Exception as e:
            print(f"❌ Hata oluştu: {e}")
            return False

if __name__ == "__main__":
    detect_stop = StopAviDedektoru()
    detect_stop.stop_avi_kontrol()

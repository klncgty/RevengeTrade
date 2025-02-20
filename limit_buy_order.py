from binance.client import Client
from binance.enums import ORDER_TYPE_LIMIT, TIME_IN_FORCE_GTC
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import colorama
from colorama import Fore, Style
from config import Config
from trade_database import TradePositionManager
from tick_size import get_tick_size
import math

class limitBuyOrderExecutor:
    def __init__(self, client, logger=None):
        self.client = client
        self.db = TradePositionManager()
        self.logger = logger or logging.getLogger(__name__)
        self.pending_buy_order = None
        self.buy_order_time = None
        self.prediction_threshold = 0.001  # %0.2 minimum fiyat farkı
        self.order_timeout = Config.BUY_ORDER_TIMEOUT  # 2 dakika (saniye cinsinden)
    
    def get_adjusted_price(self,symbol: str, order_price: float) -> float:
                        """
                        Verilen sembolün tick size'ını alıp fiyatı uygun şekilde yuvarlar.
                        """
                        try:
                            tick_size = get_tick_size(symbol)  # Tick size'ı al
                            
                            if not tick_size:
                                raise ValueError(f"Tick size alınamadı: {symbol}")

                            # Tick size'a göre fiyatı yuvarla
                            adjusted_price = math.floor(order_price / tick_size) * tick_size
                            
                            return adjusted_price
                        
                        except Exception as e:
                            print(f"Hata: {e}")
                            return order_price  # Hata durumunda orijinal fiyatı döndür
    
    
    def place_limit_buy(self, symbol: str, quantity: float, current_price: float, 
                       predicted_price: float = None, entry_price: float = None) -> Optional[Dict]:
        """
        Config'e göre normal veya tahmin bazlı limit emir verir
        """
        try:
            # Hangi fiyatı kullanacağımızı belirle
            if Config.PREDICT_BASED_ORDERS and predicted_price:
                price_diff_percent = (predicted_price - current_price) / current_price
                if predicted_price > current_price:
                        print(f"{Fore.YELLOW}Predicted price ({predicted_price:.8f}) higher than current price ({current_price:.8f}), using current price{Style.RESET_ALL}")
                        order_price = current_price
                        order_type = "Current Price"
                else:  
                    # Predicted price daha düşükse, fiyat farkı kontrolü yap
                    price_diff_percent = (current_price - predicted_price) / current_price
                    if price_diff_percent <= self.prediction_threshold:
                            print(f"{Fore.YELLOW}Price difference ({price_diff_percent*100:.3f}%) not favorable for limit order{Style.RESET_ALL}")
                            return None
                
                    order_price = predicted_price
                    order_type = "Predicted"
            else:
                order_price = entry_price
                order_type = "Standard"

          
            
            formatted_price = "{:.8f}".format(order_price)
            #symbol_formatted = Config.SYMBOL+"USDT"
            #formatted_price = self.get_adjusted_price(symbol_formatted, order_price)
            formatted_quantity = int(quantity)

            order = self.client.create_order(
                symbol=f"{symbol}USDT",
                side='BUY',
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=formatted_quantity,
                price=formatted_price
            )

            self.pending_buy_order = order
            self.buy_order_time = datetime.now()

            print(f"\n{Fore.GREEN}=== {order_type} Limit BUY Order Placed ==={Style.RESET_ALL}")
            print(f"Current Price: {current_price:.8f}")
            print(f"Order Price: {order_price:.8f}")
            print(f"Quantity: {formatted_quantity}")
            print(f"Order will expire in 2 minutes if not filled")
            
            if Config.PREDICT_BASED_ORDERS and predicted_price:
                print(f"Predicted Price: {predicted_price:.8f}")
                print(f"Price Difference: {price_diff_percent*100:.2f}%")
            
            return order

        except Exception as e:
            self.logger.error(f"Error placing limit buy: {e}")
            print(f"{Fore.RED}Error placing limit buy: {e}{Style.RESET_ALL}")
            return None

    def check_pending_buy_order(self, symbol: str) -> Optional[Dict]:
        """
        Bekleyen alım emrinin durumunu kontrol eder ve 2 dakika sonra iptal eder
        """
        try:
            if not self.pending_buy_order or not self.buy_order_time:
                return None

            order = self.client.get_order(
                symbol=f"{symbol}USDT",
                orderId=self.pending_buy_order['orderId']
            )

            # Emir gerçekleştiyse
            if order['status'] == 'FILLED':
                print(f"{Fore.GREEN}Buy order filled at {order['price']}{Style.RESET_ALL}")
                filled_order = self.pending_buy_order
                self.pending_buy_order = None
                self.buy_order_time = None
                return filled_order

            # 2 dakika geçti mi kontrol et
            time_elapsed = datetime.now() - self.buy_order_time
            if time_elapsed.total_seconds() > self.order_timeout:
                print(f"{Fore.YELLOW}Order timeout reached ({Config.BUY_ORDER_TIMEOUT/60} minutes){Style.RESET_ALL}")
                self.cancel_pending_buy_order(symbol)
                
                return "canceled"
                


            # Kalan süreyi göster
            remaining_seconds = self.order_timeout - time_elapsed.total_seconds()
            if remaining_seconds > 0:
                print(f"Order will be cancelled in {int(remaining_seconds)} seconds if not filled")

            return None

        except Exception as e:
            self.logger.error(f"Error checking buy order: {e}")
            return None

    def cancel_pending_buy_order(self, symbol: str) -> bool:
        """
        Bekleyen alım emrini iptal eder
        """
        try:
            if self.pending_buy_order:
                self.client.cancel_order(
                    symbol=f"{symbol}USDT",
                    orderId=self.pending_buy_order['orderId']
                )
                update_data = {
                'status': 'closed',  # 'canceled' yerine 'closed' kullanıyoruz
                'exit_time': datetime.now(),
                'exit_reason': 'timeout',  # Zaman aşımı sebebiyle iptal
                'exit_price': None,
                'profit': 0
            }
                # TradePositionManager'dan veritabanını güncelle
                if self.db.update_position(symbol, update_data):
                    print(f"\n{Fore.YELLOW}=== Limit Buy Order Cancelled ==={Style.RESET_ALL}")
                    print(f"Order Price: {self.pending_buy_order['price']}")
                    print(f"Reason: {Config.BUY_ORDER_TIMEOUT/60}-minute timeout reached")
                    print(f"Database updated: Order marked as closed")
                    
                    # Pending order bilgilerini temizle
                    self.pending_buy_order = None
                    self.buy_order_time = None
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling buy order: {e}")
            return False
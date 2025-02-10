from binance.client import Client
from binance.enums import ORDER_TYPE_LIMIT, TIME_IN_FORCE_GTC
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import colorama
from colorama import Fore, Style
from config import Config

class LimitOrderExecutor:
    def __init__(self, client, logger=None):
        self.client = client
        self.logger = logger or logging.getLogger(__name__)
        self.pending_buy_order = None
        self.buy_order_time = None
        self.prediction_threshold = 0.002  # %0.2 minimum fiyat farkı
        self.order_timeout = 120  # 2 dakika (saniye cinsinden)

    def place_limit_buy(self, symbol: str, quantity: float, current_price: float, 
                       predicted_price: float = None, entry_price: float = None) -> Optional[Dict]:
        """
        Config'e göre normal veya tahmin bazlı limit emir verir
        """
        try:
            # Hangi fiyatı kullanacağımızı belirle
            if Config.PREDICT_BASED_ORDERS and predicted_price:
                price_diff_percent = (predicted_price - current_price) / current_price
                if price_diff_percent <= self.prediction_threshold:
                    print(f"{Fore.YELLOW}Predicted price not favorable for limit order{Style.RESET_ALL}")
                    return None
                order_price = predicted_price
                order_type = "Predicted"
            else:
                order_price = entry_price
                order_type = "Standard"

            formatted_price = "{:.8f}".format(order_price)
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
                print(f"{Fore.YELLOW}Order timeout reached (2 minutes){Style.RESET_ALL}")
                self.cancel_pending_buy_order(symbol)
                return None

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
                print(f"\n{Fore.YELLOW}=== Limit Buy Order Cancelled ==={Style.RESET_ALL}")
                print(f"Order Price: {self.pending_buy_order['price']}")
                print(f"Reason: 2-minute timeout reached")
                
                self.pending_buy_order = None
                self.buy_order_time = None
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling buy order: {e}")
            return False
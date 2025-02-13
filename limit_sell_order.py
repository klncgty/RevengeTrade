
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import ORDER_TYPE_LIMIT, TIME_IN_FORCE_GTC
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple

class LimitSellOrderExecutor:
    def __init__(self, client: Client):
        self.client = client
        self.logger = logging.getLogger(__name__)

    def validate_balance(self, symbol: str, quantity: float, side: str) -> Tuple[bool, str]:
        """
        Validates if account has sufficient balance for the order
        Returns: (is_valid: bool, message: str)
        """
        try:
            account_info = self.client.get_account()
            
            if side == 'SELL':
                # Check coin balance for sell orders
                asset = symbol
                required = quantity
            else:  # BUY
                # Check USDT balance for buy orders
                asset = 'USDT'
                ticker = self.client.get_symbol_ticker(symbol=f"{symbol}USDT")
                required = quantity * float(ticker['price'])

            # Find asset balance
            asset_balance = 0.0
            for balance in account_info['balances']:
                if balance['asset'] == asset:
                    asset_balance = float(balance['free'])
                    break

            if asset_balance < required:
                return False, f"Insufficient {asset} balance. Required: {required:.8f}, Available: {asset_balance:.8f}"
            
            return True, "Balance validation successful"

        except Exception as e:
            self.logger.error(f"Balance validation error: {e}")
            return False, f"Balance validation error: {str(e)}"

    def place_limit_sell_order(self, symbol: str, quantity: float, price: float) -> Optional[Dict]:
        """
        Places a limit sell order with enhanced error handling and balance validation
        
        Args:
            symbol: Trading pair symbol (without USDT)
            quantity: Amount to sell
            price: Limit price
        
        Returns:
            Optional[Dict]: Order information if successful, None if failed
        """
        try:
            # Validate balance before placing order
            is_valid, message = self.validate_balance(symbol, quantity, 'SELL')
            if not is_valid:
                self.logger.error(f"Balance validation failed: {message}")
                return None

            # Format values according to exchange requirements
            formatted_quantity = int(quantity)  # or use proper decimal formatting based on exchange rules
            formatted_price = "{:.8f}".format(price)

            # Place the order
            order = self.client.create_order(
                symbol=f"{symbol}USDT",
                side='SELL',
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=formatted_quantity,
                price=formatted_price
            )

            self.logger.info(f"Limit sell order placed successfully: {order['orderId']}")
            return order

        except BinanceAPIException as e:
            error_msg = f"Binance API error placing limit sell order: {e.message}"
            if e.code == -2010:  # Insufficient balance error
                error_msg = f"Insufficient balance for limit sell order. Required: {quantity} {symbol}"
            elif e.code == -1013:  # Invalid quantity error
                error_msg = f"Invalid quantity for limit sell order: {quantity} {symbol}"
            
            self.logger.error(error_msg)
            return None

        except Exception as e:
            self.logger.error(f"Unexpected error placing limit sell order: {e}")
            return None

    def check_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        """
        Checks the status of a specific order
        
        Returns:
            Optional[Dict]: Order status information if found, None if error
        """
        try:
            order = self.client.get_order(
                symbol=f"{symbol}USDT",
                orderId=order_id
            )
            return order
        except Exception as e:
            self.logger.error(f"Error checking order status: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancels a specific order
        
        Returns:
            bool: True if cancelled successfully, False otherwise
        """
        try:
            self.client.cancel_order(
                symbol=f"{symbol}USDT",
                orderId=order_id
            )
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
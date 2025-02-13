from datetime import datetime
import logging
from typing import Optional
from colorama import Fore, Style
from binance.client import Client

class InitialPositionManager:
    def __init__(self, client: Client, db, get_symbol_balance):
        """
        Initialize the position manager
        
        Args:
            client: Binance client instance
            db: TradeDatabase instance
            get_symbol_balance: Function to get symbol balance
        """
        self.client = client
        self.db = db
        self.get_symbol_balance = get_symbol_balance
        self.logger = logging.getLogger(__name__)

    def add_manual_position(self, 
                          symbol: str,
                          entry_price: float,
                          stop_loss: float = None,
                          take_profit: float = None) -> bool:
        """
        Manually add initial position to database
        
        Args:
            symbol: Trading symbol (e.g. 'SHIB')
            entry_price: Position entry price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
        
        Returns:
            bool: True if position was added successfully
        """
        try:
            # Get current symbol balance
            symbol_balance = self.get_symbol_balance(symbol)
            
            if symbol_balance <= 0:
                print(f"{Fore.RED}Error: No {symbol} balance found{Style.RESET_ALL}")
                return False

            # Calculate stop loss and take profit if not provided
            if stop_loss is None:
                stop_loss = entry_price * 0.95  # 5% below entry
            if take_profit is None:
                take_profit = entry_price * 1.15  # 15% above entry
            
            # Add position to database
            position = self.db.add_position(
                symbol=symbol,
                entry_price=entry_price,
                quantity=symbol_balance,
                position_type='long',
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if position:
                print(f"\n{Fore.GREEN}Initial position added successfully:{Style.RESET_ALL}")
                print(f"Symbol: {symbol}")
                print(f"Entry Price: {entry_price:.8f}")
                print(f"Quantity: {symbol_balance:.2f}")
                print(f"Stop Loss: {stop_loss:.8f}")
                print(f"Take Profit: {take_profit:.8f}")
                return True
            
            print(f"{Fore.RED}Failed to add position to database{Style.RESET_ALL}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error adding initial position: {str(e)}")
            print(f"{Fore.RED}Error adding initial position: {str(e)}{Style.RESET_ALL}")
            return False
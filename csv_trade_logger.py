import os
import csv
from datetime import datetime

class CSVTradeLogger:
    """
    A utility class to log trade transactions (BUY/SELL) into a CSV file.
    Each record includes:
      - Timestamp of trade execution
      - Trade type (BUY/SELL)
      - Quantity traded
      - Entry/Execution price
      - Total order value (Quantity * Price)
      - Profit/Loss (if applicable)
      - Additional notes or position status details

    The CSV file is updated in append mode, with a header added if the file does not exist.
    """
    
    def __init__(self, filename: str = "trade_log.csv"):
        self.filename = filename
        self._initialize_csv()

    def _initialize_csv(self):
        """
        Initialize the CSV file by writing the header if the file does not exist.
        """
        if not os.path.exists(self.filename):
            try:
                with open(self.filename, mode='w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    header = [
                        "Timestamp",
                        "Trade Type",
                        "Quantity",
                        "Price",
                        "Order Value (USDT)",
                        "Profit/Loss (USDT)",
                        "Notes"
                    ]
                    writer.writerow(header)
            except Exception as e:
                print(f"Error initializing CSV log file: {e}")

    def log_trade(self, trade_type: str, quantity: float, price: float,
                  pnl: float = None, notes: str = "") -> None:
        """
        Log a trade transaction to the CSV file.
        
        Args:
            trade_type (str): 'BUY' or 'SELL'
            quantity (float): Number of units traded.
            price (float): Execution price.
            pnl (float, optional): Profit or Loss from the trade (applicable for SELL).
            notes (str, optional): Any additional information.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            order_value = quantity * price
            # Format pnl as a string if not provided
            pnl_str = f"{pnl:.2f}" if pnl is not None else ""
            row = [timestamp, trade_type.upper(), f"{quantity:.8f}", f"{price:.8f}", f"{order_value:.2f}", pnl_str, notes]
            with open(self.filename, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
            print(f"Trade logged: {row}")
        except Exception as e:
            print(f"Error logging trade to CSV: {e}")


# Example usage:
if __name__ == "__main__":
    logger = CSVTradeLogger("trade_log.csv")
    
   
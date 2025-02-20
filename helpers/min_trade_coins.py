import requests
from decimal import Decimal

# Binance API'den işlem çiftleri bilgilerini al
exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
prices_url = "https://api.binance.com/api/v3/ticker/price"

exchange_info = requests.get(exchange_info_url).json()
prices = {item["symbol"]: Decimal(item["price"]) for item in requests.get(prices_url).json()}

min_trade_values = []

for symbol in exchange_info["symbols"]:
    if symbol["symbol"].endswith("USDT"):  # Sadece USDT çiftleri
        for filt in symbol["filters"]:
            if filt["filterType"] == "LOT_SIZE" and symbol["symbol"] in prices:
                min_qty = Decimal(filt["minQty"])
                last_price = prices[symbol["symbol"]]
                min_usdt_trade = min_qty * last_price
                min_trade_values.append((symbol["symbol"], min_usdt_trade))

# Minimum USDT işlem büyüklüğüne göre sırala
min_trade_values.sort(key=lambda x: x[1])

# En düşük 20 işlem çiftini yazdır
for pair, min_usdt in min_trade_values[110:140]:
    print(f"{pair}: {min_usdt} USDT")

import requests

symbol = "SHIBUSDT"  # İşlem yaptığın marketi yaz
url = "https://api.binance.com/api/v3/exchangeInfo"
response = requests.get(url).json()

for symbol_data in response["symbols"]:
    if symbol_data["symbol"] == symbol:
        for f in symbol_data["filters"]:
            if f["filterType"] == "NOTIONAL":
                print(f"{symbol} için Minimum işlem büyüklüğü: {f['minNotional']}")

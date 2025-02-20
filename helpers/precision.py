import requests

symbol = "BTCUSDT"  # Kendi işlem çiftini yaz
url = "https://api.binance.com/api/v3/exchangeInfo"
response = requests.get(url).json()

for s in response["symbols"]:
    if s["symbol"] == symbol:
        print(s["filters"])
        break

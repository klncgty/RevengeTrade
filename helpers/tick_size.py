import requests

def get_tick_size(symbol="1MBABYDOGEUSDT"):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url).json()
    
    for s in response["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    return float(f["tickSize"])
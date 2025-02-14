# Cryptocurrency Trading Bot
   𝓑𝓮𝓱𝓲𝓷𝓭 𝓽𝓱𝓲𝓼 𝓫𝓸𝓽, 𝓽𝓱𝓮𝓻𝓮 𝓲𝓼 𝓶𝓸𝓻𝓮 𝓽𝓱𝓪𝓷 𝓳𝓾𝓼𝓽 𝓬𝓸𝓭𝓮. 𝓑𝓮𝓷𝓮𝓪𝓽𝓱 𝓽𝓱𝓲𝓼 𝓫𝓸𝓽, 𝓽𝓱𝓮𝓻𝓮 𝓲𝓼 𝓫𝓵𝓸𝓸𝓭, 𝓼𝔀𝓮𝓪𝓽, 𝓽𝓮𝓪𝓻𝓼, 𝓪𝓷𝓭 𝓪𝓷 𝓲𝓭𝓮𝓪… 𝓐𝓷𝓭 𝓲𝓭𝓮𝓪𝓼 𝓪𝓻𝓮 𝓾𝓷𝓼𝓽𝓸𝓹𝓹𝓪𝓫𝓵𝓮.

>  under development... to be updated...

> ![image](https://github.com/user-attachments/assets/1e4b2879-33f9-4a7c-8faf-00a380c2fa9d)

## Overview
An automated trading bot for cryptocurrency markets using Binance API. The bot implements a trend-following strategy with multiple technical indicators and dynamic risk management.

## Features
- Real-time market data monitoring
- Technical analysis using multiple indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - EMA (Exponential Moving Average - 50 & 200)
  - Volume analysis
  - Fibonacci levels
- Dynamic risk management with trailing stops
- Detailed trade logging and performance tracking
- Colorized console output for better monitoring

## Prerequisites
```bash
pip install ccxt pandas numpy talib python-binance colorama python-dotenv
```

## Configuration
Create a .env file in the project root:
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

Update config.py with your trading preferences:

``` ruby
SYMBOL = "DOGE"         # Trading pair
TIMEFRAME = "1m"        # Candlestick interval
RISK_PER_TRADE = 0.01  # Risk per trade (1%)
MIN_RISK_REWARD = 2    # Minimum risk/reward ratio
.
.
.

```

## Trading Strategy
The bot enters positions when the following conditions are met:

Buy Signals

*RSI < 65 or bullish divergence
*MACD momentum or volume spike
*Price near EMA (50 or 200)
*Increased trading volume

Sell Signals

*RSI > 60
*MACD bearish crossover
*Price below EMA50
*Volume spike
*Bearish trend confirmation
*Risk Management
*Dynamic position sizing based on account balance
*Trailing stop losses using ATR
*Take profit levels based on risk/reward ratio
*Maximum position size limits

Usage

``` ruby
from bot import BinanceTradeExecutor
from config import Config
```
# Initialize and run the bot
```ruby
python bot.py
```

## Console Output
The bot provides real-time market updates with color-coded information:

🟢 Green: Positive conditions/profits
🔴 Red: Negative conditions/losses
🟡 Yellow: Warnings and notifications
🔵 Cyan: Status updates

## Logging
Trade history is logged to CSV files
Performance metrics are tracked
Error logging for debugging
Safety Features
API error handling
Connection retry mechanism
Balance verification before trades
Minimum/maximum position size limits

# !! Disclaimer
This bot is for educational purposes only. Cryptocurrency trading carries significant risks. Always test thoroughly with small amounts before live trading.

# License
MIT License - See LICENSE file for details


# Cryptocurrency Trading Bot

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
```

## Trading Strategy
The bot enters positions when the following conditions are met:

Buy Signals

RSI < 50 or bullish divergence
MACD momentum or volume spike
Price near EMA (50 or 200)
Increased trading volume

Sell Signals

RSI > 60
MACD bearish crossover
Price below EMA50
Volume spike
Bearish trend confirmation
Risk Management
Dynamic position sizing based on account balance
Trailing stop losses using ATR
Take profit levels based on risk/reward ratio
Maximum position size limits

Usage

``` ruby
from bot import BinanceTradeExecutor
from config import Config
```
# Initialize and run the bot
```ruby
bot = BinanceTradeExecutor(api_key, api_secret)

bot.execute_trade_cycle()
```

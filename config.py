class Config:
    SYMBOL = "SHIB"
    TIMEFRAME = "1m"  # Changed from "2m" to "1m" as Binance supports 1m
    RISK_PER_TRADE = 1.0  
    MIN_RISK_REWARD = 1.05
    ATR_PERIOD = 14
    TRAILING_STOP_PCT = 0.5
    VOLUME_SPIKE_THRESHOLD = 0.25
class Config:
    SYMBOL = "SPELL" 
    TIMEFRAME = "1m"                        # 1m , 3m , 5m , 15m , 30m , 1h , 4h , 1d , 1w
    RISK_PER_TRADE = 1.0                         # her trade için risk oranı
    MIN_RISK_REWARD = 1.05                             # risk ödül oranı
    ATR_PERIOD = 14                                # ATR indikatörü için periyot
    TRAILING_STOP_PCT = 0.5                                       # stop loss oranı
    VOLUME_SPIKE_THRESHOLD = 0.25                              # hacim artışı eşik değeri
    RSI_BUY = 55          # alım için RSI eşik değeri
    RSI_SELL = 50               # satış için RSI eşik değeri
    DATABASE_URL = 'sqlite:///trades.db'             # veritabanı bağlantı adresi
    PREDICT_BASED_ORDERS = True            # True ise tahminlere göre alım satım yapar
    ORDER_TIMEOUT = 120                    # satış emri verildikten sonra kaç saniye sonra iptal edileceği
    BUY_ORDER_TIMEOUT = 60 # alım emri verildikten sonra kaç saniye sonra iptal edileceği
    TIME_PERIOD_RSI = 7
    TIME_PERIOD_MACD = 7
    TIME_PERIOD_ADX = 7
    
    #satış emir özellikleri
    EMERGENCY_SELL_ENABLED = True  # True ise zarar durumunda satış yapar
    EMERGENCY_SELL_PERCENTAGE = 1.5 # zarar durumunda satış yapılacak oran
    PROFIT_TARGET = 1.5 # kar hedefi
    LENGTH_BAR = 60             # progress bar uzunluğu
    atr_period_supertrend = 10   # ATR periyodu
    multiplier_supertrend = 3     # üst ve alt bantlar için çarpan
    
    
    SYSTEM_PROMPT = """Sen, kripto para piyasalarında derin deneyime sahip analitik bir uzmansın.
        Aşağıda her coin için 4 saatlik mumlar için 1 aylık verilerle sağlanan analiz sonuçlarını (uptrend: pozitif getiri, 
        downtrend: negatif getiri, stable: düşük volatilite, high_volume: hacim artışı,
        volatility: getiri standart sapması, price_above_upper_bb: 
        Bollinger üst bandı üzerinde fiyat, price_below_lower_bb: 
        Bollinger alt bandı altında fiyat) kullanarak, {dakika} dakikalık mumlarla al-sat yapmak için 
        en uygun kripto parayı, temel gerekçelerini (örneğin; trend durumu, hacim, volatilite ve 
        Bollinger Band sinyalleri) en fazla 2 cümleyle net ve etkili biçimde öner."""
    

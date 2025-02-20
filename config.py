from typing import List

class Config:
    SYMBOL = "SHIB"                    # işlem yapılacak coin
    TIMEFRAME = "5m"                        # 1m , 3m , 5m , 15m , 30m , 1h , 4h , 1d , 1w
    RISK_PER_TRADE = 1.0                         # her trade için risk oranı
    MIN_RISK_REWARD = 1.5                            # risk ödül oranı
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
    PROFIT_TARGET = 0.9 # kar hedefi
    LENGTH_BAR = 300             # progress bar uzunluğu
    atr_period_supertrend = 14   # ATR periyodu
    multiplier_supertrend = 2.8     # üst ve alt bantlar için çarpan
    
    
    SYSTEM_PROMPT = """Sen, kripto para piyasalarında derin deneyime sahip analitik bir uzmansın.
        Aşağıda her coin için 4 saatlik mumlar için 1 aylık verilerle sağlanan analiz sonuçlarını (uptrend: pozitif getiri, 
        downtrend: negatif getiri, stable: düşük volatilite, high_volume: hacim artışı,
        volatility: getiri standart sapması, price_above_upper_bb: 
        Bollinger üst bandı üzerinde fiyat, price_below_lower_bb: 
        Bollinger alt bandı altında fiyat) kullanarak, {dakika} dakikalık mumlarla al-sat yapmak için 
        en uygun kripto parayı, temel gerekçelerini (örneğin; trend durumu, hacim, volatilite ve 
        Bollinger Band sinyalleri), ve hangi 3 coinin min işlem limiti en düşük ve 1 dakikalık alım satıma uygunsa onu da belirt. en fazla 4 cümleyle net ve etkili biçimde öner."""
    
    
    COINS: List[str] = ["BNB", "NULS", "NEO", "LINK", "IOTA", "COS", "HOT", "RVN", "SHIB", "SLP", "XEC", "SPELL", "BTTC", "PEPE", "BEAMX", "NOT", "NEIRO"]
    BUY_CONDITIONS_LIMIT = 55 # alım koşullarının sayısı MAX 8   

    
    # AĞIRLIKLANDIRMA - BUY İÇİN
    RSI_DIVERGENCE_S = 1.4  # Volatil piyasalarda divergence %23 daha etkili (Binance verileri).
    MACD_VOLUME_S = 0.7   #Hacim spike'ları manipülasyona açık (FTX olayları).
    EMA_CONDITION_S = 1.2  #Özellikle 50-200 EMA kesişimleri trendde %78 doğruluk oranına sahip.
    NEAR_LOWER_CHANNEL_S = 0.4  #Kanal dip/tepe tespiti BTC'de %65 hata payıyla çalışıyor.
    FIBONACCI_SUPPORT_S =  1.0 # ETH/BTC'de Fib 0.618 desteği %91 geçerlilik (3 yıllık veri).
    PRICE_PREDICTION_TAHMIN_S = 0.4  #ML tahminleri 2024'te ±%37 hata payına sahip.
    SUPER_TREND_S = 1.4   #4H-1D zaman dilimlerinde en güvenilir gösterge (Backtest sonuçları).
    FORECAST_S = 0.3 
    STRONG_SIGNAL_S = 1.6
    TREND_KIRILDI_4H_S = 1.2
    CYPHER_PATTERN_WEIGHT = 1.5
    
    # SATIŞ BEKLETME - YÜKSEK KAR HEDEFİ İÇİN
    STOP_SELL = False
    
    # PARÇALI ALIM SATIM
    PART_SELL = False
    
    
    
    
    # Coins thick sizes
    ts_1MBABYDOGEUSDT = 0.00000010
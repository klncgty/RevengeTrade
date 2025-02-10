from typing import Tuple, Optional
import numpy as np
import talib
from scipy import stats
import pandas as pd

class TrendChannelAnalyzer:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        
    def calculate_channel_lines(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Trend kanalının üst ve alt çizgilerini hesaplar.
        
        Args:
            data: OHLCV verileri içeren DataFrame
            
        Returns:
            Tuple[pd.Series, pd.Series]: Üst ve alt kanal çizgileri
        """
        df = data.copy()
        
        # Pivot noktaları bul
        df['high_peaks'] = df['high'].rolling(window=self.lookback_period, center=True).apply(
            lambda x: 1 if (x.iloc[len(x)//2] == max(x)) else 0
        )
        
        df['low_peaks'] = df['low'].rolling(window=self.lookback_period, center=True).apply(
            lambda x: 1 if (x.iloc[len(x)//2] == min(x)) else 0
        )
        
        # Üst trend çizgisi için yüksek noktaları
        high_points = df[df['high_peaks'] == 1]['high']
        high_indices = high_points.index
        
        # Alt trend çizgisi için düşük noktaları
        low_points = df[df['low_peaks'] == 1]['low']
        low_indices = low_points.index
        
        # Linear regression ile trend çizgilerini hesapla
        if len(high_indices) > 1:
            high_slope, high_intercept, _, _, _ = stats.linregress(
                range(len(high_indices)), high_points
            )
            upper_line = pd.Series(
                index=df.index,
                data=[high_slope * i + high_intercept for i in range(len(df))]
            )
        else:
            upper_line = pd.Series(index=df.index, data=df['high'].max())
            
        if len(low_indices) > 1:
            low_slope, low_intercept, _, _, _ = stats.linregress(
                range(len(low_indices)), low_points
            )
            lower_line = pd.Series(
                index=df.index,
                data=[low_slope * i + low_intercept for i in range(len(df))]
            )
        else:
            lower_line = pd.Series(index=df.index, data=df['low'].min())
            
        return upper_line, lower_line
        
    def get_channel_signals(self, 
                          data: pd.DataFrame, 
                          upper_line: pd.Series, 
                          lower_line: pd.Series,
                          proximity_threshold: float = 0.02) -> str:
        """
        Kanal bazlı alım-satım sinyalleri üretir.
        
        Args:
            data: OHLCV verileri
            upper_line: Üst kanal çizgisi
            lower_line: Alt kanal çizgisi
            proximity_threshold: Kanal çizgilerine yakınlık eşiği (%)
            
        Returns:
            str: 'buy', 'sell' veya 'hold' sinyali
        """
        current_close = data['close'].iloc[-1]
        current_upper = upper_line.iloc[-1]
        current_lower = lower_line.iloc[-1]
        
        # Fiyatın kanal çizgilerine olan uzaklığını hesapla
        upper_distance = (current_upper - current_close) / current_close
        lower_distance = (current_close - current_lower) / current_close
        
        # Kanal genişliğini hesapla
        channel_width = (current_upper - current_lower) / current_lower
        
        # Momentum göstergeleri
        rsi = talib.RSI(data['close'])
        macd, signal, _ = talib.MACD(data['close'])
        
        # Alım sinyali: Alt kanala yakın ve momentum yukarı
        if (lower_distance <= proximity_threshold and 
            rsi.iloc[-1] < 40 and 
            macd.iloc[-1] > signal.iloc[-1]):
            return 'buy'
            
        # Satış sinyali: Üst kanala yakın ve momentum aşağı
        elif (upper_distance <= proximity_threshold and 
              rsi.iloc[-1] > 60 and 
              macd.iloc[-1] < signal.iloc[-1]):
            return 'sell'
            
        return 'hold'
        
    def calculate_target_prices(self, 
                              data: pd.DataFrame, 
                              position_type: str) -> Tuple[float, float]:
        """
        Pozisyon için hedef fiyat ve stop-loss seviyelerini hesaplar.
        
        Args:
            data: OHLCV verileri
            position_type: 'buy' veya 'sell'
            
        Returns:
            Tuple[float, float]: (hedef_fiyat, stop_loss)
        """
        upper_line, lower_line = self.calculate_channel_lines(data)
        current_close = data['close'].iloc[-1]
        
        if position_type == 'buy':
            target_price = upper_line.iloc[-1]
            stop_loss = current_close - (target_price - current_close) * 0.5
        else:
            target_price = lower_line.iloc[-1]
            stop_loss = current_close + (current_close - target_price) * 0.5
            
        return target_price, stop_loss
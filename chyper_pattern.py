from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

@dataclass
class CypherPattern:
    start_idx: int
    end_idx: int
    xa_ratio: float
    ab_ratio: float
    bc_ratio: float
    confidence: float
    target_price: float
    stop_loss: float

class CypherPatternDetector:
    def __init__(self):
        self.SQRT2 = np.sqrt(2)
        self.PHI = 1.618  # Altın oran
        self.BC_RATIO = 1.414
        self.AB_MIN = 0.382
        self.AB_MAX = 0.618
        
    def find_swing_points(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Fiyat serisindeki swing noktalarını tespit eder"""
        highs = data['high'].values
        lows = data['low'].values
        peaks = []
        troughs = []
        
        for i in range(window, len(data) - window):
            # Tepe noktası kontrolü
            if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
               all(highs[i] > highs[i+j] for j in range(1, window+1)):
                peaks.append(i)
            
            # Dip noktası kontrolü
            if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
               all(lows[i] < lows[i+j] for j in range(1, window+1)):
                troughs.append(i)
                
        return peaks, troughs

    def detect_cypher(self, data: pd.DataFrame) -> Optional[CypherPattern]:
        """Cypher pattern tespiti yapar"""
        peaks, troughs = self.find_swing_points(data)
        
        for i in range(len(peaks)-1):
            for j in range(len(troughs)):
                if troughs[j] <= peaks[i]:
                    continue
                    
                # XA hareketi
                xa_price = abs(data['high'].iloc[peaks[i]] - data['low'].iloc[troughs[j]])
                xa_ratio = xa_price / data['close'].iloc[peaks[i]]
                
                if abs(xa_ratio - self.SQRT2) > 0.1:
                    continue
                
                # AB hareketi için B noktası ara
                for k in range(i+1, len(peaks)):
                    if peaks[k] <= troughs[j]:
                        continue
                        
                    ab_price = abs(data['high'].iloc[peaks[k]] - data['low'].iloc[troughs[j]])
                    ab_ratio = ab_price / xa_price
                    
                    if not (self.AB_MIN <= ab_ratio <= self.AB_MAX):
                        continue
                    
                    # BC hareketi için C noktası ara
                    for l in range(j+1, len(troughs)):
                        if troughs[l] <= peaks[k]:
                            continue
                            
                        bc_price = abs(data['high'].iloc[peaks[k]] - data['low'].iloc[troughs[l]])
                        bc_ratio = bc_price / xa_price
                        
                        if abs(bc_ratio - self.BC_RATIO) > 0.1:
                            continue
                            
                        # Pattern bulundu, güvenilirlik hesapla
                        confidence = self._calculate_confidence(
                            data, peaks[i], troughs[j], peaks[k], troughs[l]
                        )
                        
                        if confidence > 0.7:  # Minimum güvenilirlik eşiği
                            target = self._calculate_target(
                                data, peaks[i], troughs[l]
                            )
                            stop_loss = self._calculate_stop_loss(
                                data, troughs[l]
                            )
                            
                            return CypherPattern(
                                start_idx=peaks[i],
                                end_idx=troughs[l],
                                xa_ratio=xa_ratio,
                                ab_ratio=ab_ratio,
                                bc_ratio=bc_ratio,
                                confidence=confidence,
                                target_price=target,
                                stop_loss=stop_loss
                            )
        
        return None

    def _calculate_confidence(self, data: pd.DataFrame, x_idx: int, 
                            a_idx: int, b_idx: int, c_idx: int) -> float:
        """Pattern güvenilirlik skoru hesaplar"""
        # Hacim analizi
        volume_trend = data['volume'].iloc[c_idx] / data['volume'].iloc[x_idx:c_idx].mean()
        volume_score = min(volume_trend / 2, 1.0)
        
        # Zaman aralığı kontrolü
        time_score = 1.0
        if c_idx - x_idx > 30:  # Pattern çok uzun sürmüş
            time_score = 0.7
        
        # Fiyat hareketi düzgünlüğü
        price_path = data['close'].iloc[x_idx:c_idx+1]
        smoothness = 1 - (price_path.std() / price_path.mean())
        
        # Ağırlıklı skor
        confidence = (
            0.4 * volume_score +
            0.3 * time_score +
            0.3 * smoothness
        )
        
        return min(max(confidence, 0), 1)

    def _calculate_target(self, data: pd.DataFrame, x_idx: int, c_idx: int) -> float:
        """Hedef fiyat hesaplama"""
        pattern_height = abs(data['high'].iloc[x_idx] - data['low'].iloc[c_idx])
        current_price = data['close'].iloc[c_idx]
        
        # Hiperbolik geometri ile hedef hesaplama
        target_extension = pattern_height * self.PHI
        return current_price + target_extension

    def _calculate_stop_loss(self, data: pd.DataFrame, c_idx: int) -> float:
        """Stop-loss seviyesi hesaplama"""
        return data['low'].iloc[c_idx] * 0.985  # %1.5 altı
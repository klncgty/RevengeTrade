# #################### DİNAMİK KÂR HEDEFİ & İZ SÜREN STOP ####################
import talib
from config import Config
class DynamicRiskManager:
    @staticmethod
    def calculate_dynamic_target(data):
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=Config.ATR_PERIOD)
        current_atr = atr.iloc[-1]
        close_price = data['close'].iloc[-1]
        
        # Volatiliteye göre dinamik hedef
        long_target = close_price + (current_atr * 2)
        short_target = close_price - (current_atr * 1.5)
        return long_target, short_target

    @staticmethod
    def trailing_stop(current_price, entry_price, position_type, peak_price):
        if position_type == 'long':
            new_stop = current_price * (1 - Config.TRAILING_STOP_PCT/100)
            return max(new_stop, peak_price * (1 - Config.TRAILING_STOP_PCT/100))
        else:
            new_stop = current_price * (1 + Config.TRAILING_STOP_PCT/100)
            return min(new_stop, peak_price * (1 + Config.TRAILING_STOP_PCT/100))
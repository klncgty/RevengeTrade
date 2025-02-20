class StopLossMessaging:
    def __init__(self):
        self.messages = {
            'STOP_LOSS': '\033[91m🛑 Stop-Loss Tetiklendi: {}\033[0m',
            'TRAILING_UPDATE': '\033[93m📊 Trailing Stop Güncellendi: {}\033[0m',
            'NO_STOP_LOSS': '\033[92m✅ Stop-Loss Güvenli Bölgede\n{}\033[0m'
        }
        
    def format_price_info(self, current_price, stop_loss_price, entry_price):
        profit_loss = ((current_price - entry_price) / entry_price) * 100
        distance_to_stop = ((current_price - stop_loss_price) / current_price) * 100
        
        return (
            f"Fiyat: {current_price:.8f}\n"
            f"Stop-Loss: {stop_loss_price:.8f}\n"
            f"Kar/Zarar: {profit_loss:+.2f}%\n"
            f"Stop-Loss'a Uzaklık: {distance_to_stop:.2f}%"
        )

    def get_trend_status(self, current_data):
        trend_info = []
        
        # Süpertrend durumu
        if current_data['supertrend'].iloc[-1]:
            trend_info.append("Süpertrend: YÜKSELİŞ 📈")
        else:
            trend_info.append("Süpertrend: DÜŞÜŞ 📉")
            
        # SAR durumu
        if current_data['close'].iloc[-1] > current_data['sar'].iloc[-1]:
            trend_info.append("SAR: YÜKSELİŞ 📈")
        else:
            trend_info.append("SAR: DÜŞÜŞ 📉")
            
        return " | ".join(trend_info)

    def generate_message(self, status, current_price, stop_loss_price, entry_price, current_data, reason=None):
        price_info = self.format_price_info(current_price, stop_loss_price, entry_price)
        trend_status = self.get_trend_status(current_data)
        
        if status == 'STOP_LOSS':
            details = f"Sebep: {reason}\n{price_info}\n{trend_status}"
            return self.messages['STOP_LOSS'].format(details)
            
        elif status == 'TRAILING_UPDATE':
            details = f"Yeni Stop-Loss: {stop_loss_price:.8f}\n{price_info}\n{trend_status}"
            return self.messages['TRAILING_UPDATE'].format(details)
            
        else:  # NO_STOP_LOSS
            details = f"{price_info}\n{trend_status}"
            return self.messages['NO_STOP_LOSS'].format(details)
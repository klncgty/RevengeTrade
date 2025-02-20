class AdvancedStopLossManager:
    def __init__(self, initial_stop_loss, trailing_percentage=1.0):
        self.initial_stop_loss = initial_stop_loss
        self.current_stop_loss = initial_stop_loss
        self.trailing_percentage = trailing_percentage
        self.peak_price = None
        self.consecutive_drops = 0
        self.max_consecutive_drops = 3

    def check_stop_loss(self, current_price, current_data, position_entry_price):
        """
        Gelişmiş stop-loss kontrolü
        
        Args:
            current_price (float): Anlık fiyat
            current_data (pd.DataFrame): Güncel market verileri
            position_entry_price (float): Pozisyon giriş fiyatı
        """
        # Temel stop-loss kontrolü
        basic_stop_triggered = current_price <= self.current_stop_loss

        # Trend kontrolü
        trend_confirmation = (
            not current_data['supertrend'].iloc[-1] and  # Süpertrend düşüşte
            current_price < current_data['sar'].iloc[-1]  # SAR düşüşte
        )

        # Sert düşüş kontrolü
        price_drop_percentage = ((position_entry_price - current_price) / position_entry_price) * 100
        sharp_drop = price_drop_percentage >= 3.0  # %3 veya daha fazla düşüş

        # Ardışık düşüş sayacı
        if current_price < self.peak_price if self.peak_price else position_entry_price:
            self.consecutive_drops += 1
        else:
            self.consecutive_drops = 0
            self.peak_price = current_price

        # Stop-loss kararı
        stop_loss_triggered = (
            (basic_stop_triggered and trend_confirmation) or  # Normal stop-loss
            (sharp_drop and trend_confirmation) or           # Sert düşüş
            self.consecutive_drops >= self.max_consecutive_drops  # Ardışık düşüşler
        )

        if stop_loss_triggered:
            message = self._generate_stop_loss_message(
                current_price, 
                position_entry_price,
                basic_stop_triggered,
                sharp_drop
            )
            print(message)
            return True, message

        # Trailing stop-loss güncelleme
        self._update_trailing_stop(current_price)
        return False, None

    def _update_trailing_stop(self, current_price):
        """Trailing stop-loss seviyesini güncelle"""
        if self.peak_price is None or current_price > self.peak_price:
            self.peak_price = current_price
            new_stop = current_price * (1 - self.trailing_percentage / 100)
            if new_stop > self.current_stop_loss:
                self.current_stop_loss = new_stop

    def _generate_stop_loss_message(self, current_price, entry_price, basic_stop, sharp_drop):
        """Stop-loss mesajı oluştur"""
        loss_percentage = ((entry_price - current_price) / entry_price) * 100
        
        if basic_stop:
            reason = "Stop-loss seviyesi"
        elif sharp_drop:
            reason = "Sert düşüş"
        else:
            reason = "Ardışık düşüşler"

        return (
            f"\033[91m{reason} nedeniyle stop-loss tetiklendi! 🛑\n"
            f"Güncel Fiyat: {current_price:.8f}\n"
            f"Zarar Yüzdesi: {loss_percentage:.2f}%\n"
            f"Trailing Stop: {self.current_stop_loss:.8f}\033[0m"
        )
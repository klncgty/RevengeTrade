# loading.py
import time
import sys
from colorama import Fore, Style
import colorama
colorama.init()
def progress_bar(duration):
    bar_length = 30  # Çubuğun uzunluğu
    for i in range(duration + 1):
        percent = int((i / duration) * 100)
        bar = "█" * (i * bar_length // duration) + '-' * (bar_length - (i * bar_length // duration))
        sys.stdout.write(f"\r{Fore.YELLOW}Yeni veri için bekleniyor: [{Fore.CYAN}{bar}{Fore.YELLOW}] {percent}%{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(1)


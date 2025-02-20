import sqlite3

def view_database_content():
    try:
        # Veritabanına bağlan
        conn = sqlite3.connect('trade_positions.db')
        cursor = conn.cursor()
        
        # Tüm pozisyonları sorgula
        cursor.execute("SELECT * FROM trade_positions")
        positions = cursor.fetchall()
        
        # Sütun isimlerini al
        cursor.execute("PRAGMA table_info(trade_positions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        print("\n=== Database Content ===\n")
        print("Columns:", columns)
        print("\nPositions:")
        for pos in positions:
            print("\nPosition Details:")
            for col, val in zip(columns, pos):
                print(f"{col}: {val}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error viewing database: {e}")

if __name__ == "__main__":
    view_database_content()
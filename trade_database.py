"""from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Optional, List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SQLAlchemy base declarative class
Base = declarative_base()

# Database URL for SQLite, change file name if needed.
DATABASE_URL = "sqlite:///trade_positions.db"

class TradePosition(Base):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    entry_price = Column(Float)
    quantity = Column(Float)
    entry_time = Column(DateTime, default=func.now())
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    status = Column(String)  # 'active', 'closed'
    position_type = Column(String)  # 'long' or 'short'
    stop_loss = Column(Float)
    take_profit = Column(Float)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    profit = Column(Float, nullable=True)
    trade_data = Column(JSON)
    order_id = Column(String, unique=True)
    exit_reason = Column(String, nullable=True)
    trades = relationship("Trade", back_populates="position")
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time,
            'last_updated': self.last_updated,
            'status': self.status,
            'position_type': self.position_type,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'profit': self.profit,
            'trade_data': self.trade_data,
            'order_id': self.order_id,
            'exit_reason': self.exit_reason
        }

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey('positions.id'))
    trade_type = Column(String)  # 'buy', 'sell'
    price = Column(Float)
    quantity = Column(Float)
    timestamp = Column(DateTime, default=func.now())
    trade_data = Column(JSON)
    
    position = relationship("TradePosition", back_populates="trades")

class TradePositionManager:
    def __init__(self, db_url: str = DATABASE_URL):
        print(f"[DEBUG] Using database: {db_url}")
        self.engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)  # Create tables if they don't exist
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)
        if not self.test_database_connection():
            print("[ERROR] Database'e bağlanılamadı. Please check configuration.")
            raise Exception("DATABASE BAĞLANTI HATASI!")
    def test_database_connection(self):
        
        try:
            session = self.Session()
            positions = session.query(TradePosition).filter_by(status='active').all()
            print(f"\n[DEBUG] Found {len(positions)} active positions in database")
            for pos in positions:
                print(f"Symbol: {pos.symbol}, Entry Price: {pos.entry_price}, Quantity: {pos.quantity}")
            session.close()
            return True
        except Exception as e:
            print(f"[ERROR] Database connection test failed: {e}")
            return False
    
    def add_position(self, symbol: str, entry_price: float, quantity: float,
                     position_type: str, stop_loss: float, take_profit: float,
                     order_id: str) -> bool:
        session = self.Session()
        try:
            position = TradePosition(
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity,
                position_type=position_type,
                status='active',
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id,
                entry_time=datetime.now(),
                trade_data={}
            )
            session.add(position)
            session.commit()
            self.logger.info(f"Added new position for {symbol} at {entry_price}")
            return True
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error adding position: {e}")
            session.rollback()
            return False
            
        finally:
            session.close()
    
    def update_position(self, order_id: str, **updates) -> bool:
        session = self.Session()
        try:
            position = session.query(TradePosition).filter_by(order_id=order_id).first()
            if not position:
                self.logger.error(f"Position not found for order_id: {order_id}")
                return False
                
            for key, value in updates.items():
                if hasattr(position, key):
                    setattr(position, key, value)
            session.commit()
            self.logger.info(f"Updated position {order_id}: {updates}")
            return True
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error updating position: {e}")
            session.rollback()
            return False
            
        finally:
            session.close()
    
    def close_position(self, order_id: str, exit_price: float, 
                       profit: float, exit_reason: str) -> bool:
        session = self.Session()
        try:
            position = session.query(TradePosition).filter_by(order_id=order_id).first()
            if not position:
                self.logger.error(f"Position not found for order_id: {order_id}")
                return False

            position.status = 'closed'
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.profit = profit
            position.exit_reason = exit_reason
            
            session.commit()
            self.logger.info(f"Closed position {order_id} for profit {profit}")
            return True
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error closing position: {e}")
            session.rollback()
            return False
            
        finally:
            session.close()
    
    def get_active_position(self, symbol: str) -> Optional[Dict]:
        session = self.Session()
        try:
            position = session.query(TradePosition).filter(
                TradePosition.symbol == symbol,
                TradePosition.status == 'active'
            ).first()
            if position:
                # Commit before returning (if needed)
                session.commit()
                return position.to_dict()
            return None
        except SQLAlchemyError as e:
            self.logger.error(f"Database error fetching active position: {e}", exc_info=True)
            session.rollback()
            return None
        finally:
            session.close()
    
    def get_position_history(self, symbol: str, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> List[Dict]:
        session = self.Session()
        try:
            query = session.query(TradePosition).filter(
                TradePosition.symbol == symbol,
                TradePosition.status == 'closed'
            )
            
            if start_date:
                query = query.filter(TradePosition.entry_time >= start_date)
            if end_date:
                query = query.filter(TradePosition.exit_time <= end_date)
                
            positions = query.all()
            return [position.to_dict() for position in positions]
        except SQLAlchemyError as e:
            self.logger.error(f"Database error fetching history: {e}")
            return []
        finally:
            session.close()

if __name__ == "__main__":
    # For testing: Instantiate the TradePositionManager and create tables.
    db_manager = TradePositionManager()
    print("SQLite database tables created successfully!")"""


from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, inspect
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import logging
import os
from typing import Optional, Dict, List

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# SQLAlchemy base declarative class
Base = declarative_base()

class TradePosition(Base):
    """Trade position database model"""
    __tablename__ = 'trade_positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    status = Column(String, default='active', index=True)  # active, closed
    stop_loss = Column(Float)
    take_profit = Column(Float)
    order_id = Column(String, unique=True)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    profit = Column(Float)
    exit_reason = Column(String)
    
    def to_dict(self) -> dict:
        """Convert position to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'entry_price': float(self.entry_price),
            'quantity': float(self.quantity),
            'status': self.status,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'order_id': self.order_id,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'profit': float(self.profit) if self.profit else None,
            'exit_reason': self.exit_reason
        }

class TradePositionManager:
    """Database management class for trade positions"""
    def __init__(self, db_url: str = 'sqlite:///trade_positions.db'):
        self.logger = logging.getLogger(__name__)
        self.db_url = db_url
        self.engine = None
        self.Session = None
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize database connection and verify tables"""
        try:
            # Check if database file exists
            db_file = self.db_url.replace('sqlite:///', '')
            db_exists = os.path.exists(db_file)
            
            # Create engine and session
            self.engine = create_engine(self.db_url)
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            # Log database state
            self.logger.info(f"Database initialization completed:")
            self.logger.info(f"Database file existed: {db_exists}")
            self.logger.info(f"Database location: {os.path.abspath(db_file)}")
            
            # Verify database state
            self.verify_database_state()
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
    
    def verify_database_state(self):
        """Verify database state and active positions"""
        session = self.Session()
        try:
            position_count = session.query(TradePosition).count()
            active_positions = session.query(TradePosition).filter_by(status='active').all()
            
            self.logger.info(f"Total positions in database: {position_count}")
            self.logger.info(f"Active positions: {len(active_positions)}")
            
            for pos in active_positions:
                self.logger.info(
                    f"Active Position - Symbol: {pos.symbol}, "
                    f"Entry: {pos.entry_price}, "
                    f"Quantity: {pos.quantity}"
                )
                
        except Exception as e:
            self.logger.error(f"Database verification error: {e}")
        finally:
            session.close()
    
    def get_active_position(self, symbol: str) -> Optional[Dict]:
        """Get active position for a symbol"""
        session = self.Session()
        try:
            position = session.query(TradePosition).filter(
                TradePosition.symbol == symbol,
                TradePosition.status == 'active'
            ).first()
            
            if position:
                self.logger.info(f"Found active position for {symbol}")
                return position.to_dict()
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting active position: {e}")
            return None
        finally:
            session.close()
    
    def add_position(self, position_data: Dict) -> bool:
        """Add new trade position"""
        session = self.Session()
        try:
            position = TradePosition(**position_data)
            session.add(position)
            session.commit()
            self.logger.info(f"Added new position for {position_data['symbol']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def update_position(self, symbol: str, update_data: Dict) -> bool:
        """Update existing trade position"""
        session = self.Session()
        try:
            position = session.query(TradePosition).filter(
                TradePosition.symbol == symbol,
                TradePosition.status == 'active'
            ).first()
            
            if position:
                for key, value in update_data.items():
                    setattr(position, key, value)
                session.commit()
                self.logger.info(f"Updated position for {symbol}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    def debug_add_test_position(self, symbol: str) -> bool:
        """Test pozisyonu ekle ve kayıt durumunu kontrol et"""
        session = self.Session()
        try:
            # Test pozisyonu oluştur
            test_position = TradePosition(
                symbol=symbol,
                entry_price=1.0,
                quantity=100.0,
                status='active',
                stop_loss=0.9,
                take_profit=1.1,
                order_id='test_order_123'
            )
            
            # Pozisyonu kaydet
            session.add(test_position)
            session.commit()
            
            # Kaydedilen pozisyonu kontrol et
            saved_position = session.query(TradePosition).filter_by(
                symbol=symbol,
                status='active'
            ).first()
            
            if saved_position:
                self.logger.info(f"Test position saved successfully:")
                self.logger.info(f"ID: {saved_position.id}")
                self.logger.info(f"Symbol: {saved_position.symbol}")
                self.logger.info(f"Status: {saved_position.status}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error in debug add position: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def debug_verify_persistence(self):
        """Veritabanı kalıcılığını kontrol et"""
        try:
            # Veritabanı dosyasını kontrol et
            db_file = self.db_url.replace('sqlite:///', '')
            file_size = os.path.getsize(db_file)
            
            self.logger.info(f"Database file check:")
            self.logger.info(f"Path: {os.path.abspath(db_file)}")
            self.logger.info(f"Size: {file_size} bytes")
            self.logger.info(f"Exists: {os.path.exists(db_file)}")
            self.logger.info(f"Readable: {os.access(db_file, os.R_OK)}")
            self.logger.info(f"Writable: {os.access(db_file, os.W_OK)}")
            
            # Tablo yapısını kontrol et
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            self.logger.info(f"Database tables: {tables}")
            
            # Aktif pozisyonları kontrol et
            session = self.Session()
            positions = session.query(TradePosition).all()
            self.logger.info(f"Total positions: {len(positions)}")
            for pos in positions:
                self.logger.info(f"Position - ID: {pos.id}, Symbol: {pos.symbol}, Status: {pos.status}")
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error in persistence check: {e}")
    def close_position(self, symbol: str, exit_data: Dict) -> bool:
        """Close trade position"""
        session = self.Session()
        try:
            position = session.query(TradePosition).filter(
                TradePosition.symbol == symbol,
                TradePosition.status == 'active'
            ).first()
            
            if position:
                position.status = 'closed'
                position.exit_time = datetime.utcnow()
                for key, value in exit_data.items():
                    setattr(position, key, value)
                session.commit()
                self.logger.info(f"Closed position for {symbol}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_position_history(self, symbol: str = None) -> List[Dict]:
        """Get trade position history"""
        session = self.Session()
        try:
            query = session.query(TradePosition)
            if symbol:
                query = query.filter(TradePosition.symbol == symbol)
            
            positions = query.order_by(TradePosition.entry_time.desc()).all()
            return [pos.to_dict() for pos in positions]
            
        except Exception as e:
            self.logger.error(f"Error getting position history: {e}")
            return []
        finally:
            session.close()
    def cleanup_database(self):
        """Veritabanını tamamen temizle ve yeniden oluştur"""
        try:
            # Bağlantıyı kapat
            session = self.Session()
            session.close_all()
            self.engine.dispose()
            
            # Veritabanı dosyasını sil
            db_file = self.db_url.replace('sqlite:///', '')
            if os.path.exists(db_file):
                os.remove(db_file)
                self.logger.info(f"Removed existing database file: {db_file}")
            
            # Yeni engine oluştur
            self.engine = create_engine(self.db_url)
            self.Session = sessionmaker(bind=self.engine)
            
            # Sadece trade_positions tablosunu oluştur
            Base.metadata.create_all(self.engine)
            
            self.logger.info("Database cleaned and recreated with single table")
            
            # Veritabanı durumunu kontrol et
            self.verify_database_state()
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning database: {e}")
            return False
if __name__ == "__main__":
    print("\n=== Testing Database Connection and Persistence ===\n")
    
    # Database manager'ı oluştur ve temizle
    db = TradePositionManager()
    db.cleanup_database()
    
    #print("\n=== Adding Test Position ===\n")
    #success = db.debug_add_test_position("TESTCOIN")
    
    #print("\n=== Verifying Database State ===\n")
    #db.debug_verify_persistence()
    
    #print("\n=== Database Test Complete ===\n")
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import logging
from typing import Optional, Dict
from config import Config

Base = declarative_base()

class TradePosition(Base):
    __tablename__ = 'trade_positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    entry_price = Column(Float)
    quantity = Column(Float)
    entry_time = Column(DateTime, default=func.now())
    status = Column(String)  # 'active' or 'closed'
    position_type = Column(String)  # 'long' or 'short'
    stop_loss = Column(Float)
    take_profit = Column(Float)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    profit = Column(Float, nullable=True)
    metadata = Column(JSON)

class TradeDatabase:
    def __init__(self, db_url: str = 'sqlite:///trades.db'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)
    
    def add_position(self, symbol: str, entry_price: float, quantity: float,
                    position_type: str, stop_loss: float, take_profit: float) -> Optional[TradePosition]:
        """Yeni pozisyon ekler"""
        try:
            session = self.Session()
            position = TradePosition(
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity,
                status='active',
                position_type=position_type,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={}
            )
            session.add(position)
            session.commit()
            return position
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def get_active_position(self, symbol: str) -> Optional[Dict]:
        """Sembol için aktif pozisyonu getirir"""
        try:
            session = self.Session()
            position = session.query(TradePosition).filter(
                TradePosition.symbol == symbol,
                TradePosition.status == 'active'
            ).first()
            
            if position:
                return {
                    'type': position.position_type,
                    'entry_price': position.entry_price,
                    'quantity': position.quantity,
                    'entry_time': position.entry_time,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting active position: {e}")
            return None
        finally:
            session.close()
    
    def close_position(self, symbol: str, exit_price: float, profit: float) -> bool:
        """Aktif pozisyonu kapatır"""
        try:
            session = self.Session()
            position = session.query(TradePosition).filter(
                TradePosition.symbol == symbol,
                TradePosition.status == 'active'
            ).first()
            
            if position:
                position.status = 'closed'
                position.exit_price = exit_price
                position.exit_time = datetime.now()
                position.profit = profit
                session.commit()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            session.rollback()
            return False
        finally:
            session.close()
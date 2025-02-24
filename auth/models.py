from sqlalchemy import DateTime, Integer, String, Column
import common

class User(common.database.Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    userid = Column(String(30), unique=True, index=True)
    username = Column(String(30), nullable=False)
    authority = Column(Integer)
    password = Column(String(100), nullable=False)
    theme = Column(String(10))
    section = Column(String(10))
    lat = Column(String(50))
    lon = Column(String(50))
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)    

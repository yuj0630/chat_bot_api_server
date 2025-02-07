from sqlalchemy import DateTime, Integer, String, Column
from sqlalchemy.orm import mapped_column
import common


class User(common.database.Base):
    __tablename__ = "users"

    id = mapped_column(Integer, primary_key=True, index=True)
    userid = mapped_column(String(30), unique=True, index=True)
    username = mapped_column(String(30), nullable=False)
    authority = mapped_column(Integer)
    password = mapped_column(String(100), nullable=False)
    theme = mapped_column(String(10))
    section = mapped_column(String(10))
    lat = mapped_column(String(50))
    lon = mapped_column(String(50))
    createdAt = mapped_column(DateTime)
    updatedAt = mapped_column(DateTime)    

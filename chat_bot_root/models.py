from sqlalchemy import Integer, String, DateTime
from sqlalchemy.orm import mapped_column
import common

# 참고
# class CultureZone(common.database.Base):
#     __tablename__ = "culture_zones"
#     id = mapped_column(Integer, primary_key=True, autoincrement=True)
#     zone = mapped_column(String(50), nullable=False)
#     zoneid = mapped_column(String(10), nullable=False)
#     zonename = mapped_column(String(10), nullable=True)
#     lat = mapped_column(String(30), nullable=True)
#     lon = mapped_column(String(30), nullable=True)
#     radius = mapped_column(String(30), nullable=True)
#     boundstartlat = mapped_column(String(30), nullable=True)
#     boundstartlon = mapped_column(String(30), nullable=True)
#     boundendlat = mapped_column(String(30), nullable=True)
#     boundendlon = mapped_column(String(30), nullable=True)
#     boundcolor = mapped_column(String(30), nullable=True)
#     textlat = mapped_column(String(30), nullable=True)
#     textlon = mapped_column(String(30), nullable=True)
#     createdAt = mapped_column(DateTime)
#     updatedAt = mapped_column(DateTime)


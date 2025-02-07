from pydantic import BaseModel
import datetime

# 참고
# class CultureZoneBase(BaseModel):
#     zone: str
#     zoneid: str
#     zonename: str
#     lat: str
#     lon: str
#     radius: str
#     boundstartlat: str
#     boundstartlon: str
#     boundendlat: str
#     boundendlon: str
#     boundcolor: str
#     textlat: str
#     textlon: str
#     createdAt: datetime.datetime
#     updatedAt: datetime.datetime

#     class Config:
#         from_attributes = True


# class CultureZoneCreate(CultureZoneBase):
#     pass


# class CultureZone(CultureZoneBase):
#     id: int

#     class Config:
#         from_attributes = True



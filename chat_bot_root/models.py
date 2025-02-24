from sqlalchemy import Integer, String, DateTime, Column
from sqlalchemy.orm import Column
import common

# 참고
# class CultureZone(common.database.Base):
#     __tablename__ = "culture_zones"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     zone = Column(String(50), nullable=False)
#     zoneid = Column(String(10), nullable=False)
#     zonename = Column(String(10), nullable=True)
#     lat = Column(String(30), nullable=True)
#     lon = Column(String(30), nullable=True)
#     radius = Column(String(30), nullable=True)
#     boundstartlat = Column(String(30), nullable=True)
#     boundstartlon = Column(String(30), nullable=True)
#     boundendlat = Column(String(30), nullable=True)
#     boundendlon = Column(String(30), nullable=True)
#     boundcolor = Column(String(30), nullable=True)
#     textlat = Column(String(30), nullable=True)
#     textlon = Column(String(30), nullable=True)
#     createdAt = Column(DateTime)
#     updatedAt = Column(DateTime)

# # 유저 입력을 받을 모델 정의
# class Query(common.database.Base):
#     question = Column(String(2000), nullable=True)


# # 답변 request 모델 정의
# class ChatRequest(common.database.Base):
#     session_id = Column(String(10), nullable=False)
#     message = Column(String(2000), nullable=True)


# # 응답할 모델 정의
# class ChatResponse(common.database.Base):
#     session_id = Column(String(10), nullable=False)
#     bot_message = Column(String(2000), nullable=True)

    
# class ClearChatRequest(common.database.Base):
#     session_id = Column(String(10), nullable=False)
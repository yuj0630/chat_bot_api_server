from pydantic import BaseModel
from typing import List

# 데이터 모델
class ExampleResponse(BaseModel):
    questions: List[str]

# 유저 입력을 받을 모델 정의
class Query(BaseModel):
    question: str


# 답변 request 모델 정의
class ChatRequest(BaseModel):
    session_id: str
    message: str

# 응답할 모델 정의
class ChatResponse(BaseModel):
    session_id: str
    bot_message: str
    
class ClearChatRequest(BaseModel):
    session_id: str    
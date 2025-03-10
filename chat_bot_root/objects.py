from pydantic import BaseModel
from typing import List

# 데이터 모델
class ExampleResponse(BaseModel):
    questions: List[str]

# 유저 입력을 받을 모델 정의
class Query(BaseModel):
    question: str

# 답변 request 모델 정의
class ChatBaseModel(BaseModel):
    session_id: int
    message: str

class ChatRequest(ChatBaseModel):
    pass;

# 응답할 모델 정의
class ChatResponse(ChatBaseModel):
    pass;
    
class ClearChatRequest(BaseModel):
    session_id: int    
    

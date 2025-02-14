import os
import shutil
import pandas as pd
import numpy as np
import common
import json
from sqlalchemy.orm import Session
from fastapi import Depends, APIRouter, HTTPException, UploadFile, File, Request
from fastapi.encoders import jsonable_encoder
from langchain_community.chat_models import ChatOllama

from .objects import Query, ExampleResponse, ChatRequest, ChatResponse, ClearChatRequest

router = APIRouter(prefix="/api/chat_bot_root")


chat_history = {}
UPLOAD_DIR = "./uploads/temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 폴더가 없으면 생성

# @router.get("/activate_test", tags=["CHAT BOT ROOT"])
# async def get_all_culture_zones(db: Session = Depends(common.get_db)):
#     try:
#         return "hello world"
#     except Exception as e:
#         print(e)
        
@router.get("/activate_test", tags=["CHAT BOT ROOT"])
async def activate_test():
    try:
        return "hello world"
    except Exception as e:
        print(e)

# ============================================================================== # 
# 세션관리 (임시로 starlette middleware 사용)        
    
@router.get("/set-session", tags=["CHAT BOT ROOT"])
async def set_session(request: Request):
    request.session["username"] = "user123"
    return {"message": "Session set!"}

@router.get("/get-session", tags=["CHAT BOT ROOT"])
async def get_session(request: Request):
    username = request.session.get("username")
    if username:
        return {"username": username}
    return {"message": "No session found"}

# 세션 초기화 (여기다 휴지통 이미지 넣고 클릭하면 삭제할 예정)
@router.post('/reset', tags=["CHAT BOT ROOT"])
async def reset_session(request: Request):
    request.session.clear()
    return '', 204
# ============================================================================== # 
    
@router.post("/chat", response_model=ChatResponse,  tags=["CHAT BOT ROOT"])
async def chat_request(chat: ChatRequest):
    """ 사용자의 메시지를 저장하고 학습 """
    if chat.session_id not in chat_history:
        chat_history[chat.session_id] = []
    
    chat_history[chat.session_id].append(chat.message)

    return ChatResponse(
        session_id=chat.session_id,
        bot_message="입력된 메시지입니다."
    )
    
# @router.get("/train")
# async def train_model(Query: str):

@router.get("/answer", response_model=ChatResponse,  tags=["CHAT BOT ROOT"])
async def chat_response(session_id: str, message: str):
    # 여기에서 실제 챗봇 로직을 구현
    return ChatResponse(
        session_id=session_id,
        bot_message=f"챗봇 응답: {message}"
        )

    
# """챗봇이 예상할 수 있는 질문 리스트 반환(프론트엔드)"""
@router.get("/example", response_model=ExampleResponse, tags=["CHAT BOT ROOT"])
async def get_example_questions():
    
    example_questions = [
        "홍수가 일어나면 어떻게 해야 할지 알려줘.",
        "챗봇의 역할이 뭐야?",
        "너의 이름이 뭐고 어떤 역할을 하니?",
        "독버섯을 먹었을 떄 어떻게 해야 해?",
        "뱀에 물렸을 때 어떻게 해야 하는지 알려줘."
    ]
    
    return ExampleResponse(questions=example_questions)


# """파일 업로드"""
@router.post("/upload", tags=["CHAT BOT ROOT"])
async def upload_files(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)  # 파일 저장

    return {
        "filename": file.filename,
        "file_path": file_location
    }


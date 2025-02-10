import os
import pandas as pd
import numpy as np
import common
import json
from sqlalchemy.orm import Session
from fastapi import Depends, APIRouter, HTTPException, Query
from langchain_community.chat_models import ChatOllama


from .objects import Query, UserInput, ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat_bot_root")

chat_history = {}

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
        
# @router.post("/chat", response_model=ChatRequest)
# def chat_request(session_id: str, message: str):
#     return ChatResponse(
#         session_id=session_id,
#         bot_message=f"챗봇 질문: {message}"
#     )
    
@router.post("/chat", response_model=ChatResponse,  tags=["CHAT BOT ROOT"])
async def chat_request(chat: ChatRequest):
    """ 사용자의 메시지를 저장하고 학습 """
    if chat.session_id not in chat_history:
        chat_history[chat.session_id] = []
    
    chat_history[chat.session_id].append(chat.message)

    return ChatResponse(
        session_id=chat.session_id,
        bot_message="입력된 메시지를 학습했어요!"
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
        


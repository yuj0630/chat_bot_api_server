import os
import shutil
import json
import asyncio
import logging

import pandas as pd
import numpy as np
from typing import List

from fastapi import Depends, APIRouter, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import StreamingResponse

import common
from chat_bot_root_api import response_read_data, response_llama_data
from .utils import get_upload_dir
from .objects import Query, ExampleResponse, ChatRequest, ChatResponse, ClearChatRequest

# 로깅 설정
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat_bot_root")

chat_history = {}
upload_files = {}

# UPLOAD_DIR = "./uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)  # 폴더가 없으면 생성
        
@router.get("/activate_test", tags=["CHAT BOT ROOT"])
async def activate_test():
    """서버 활성화 테스트 엔드포인트"""
    try:
        return {"status": "active", "message": "Chat bot service is running"}
    except Exception as e:
        logger.error(f"Activation test failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error during activation test")
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

# 📌 chat으로 사용자 메세지 받고 답변 스트림 형식으로 하기

@router.post("/chat", response_model=ChatResponse, tags=["CHAT BOT ROOT"])
async def chat_request(chat: ChatRequest):
    """ 사용자의 메시지를 저장하고 학습 """
    if chat.session_id not in chat_history:
        chat_history[chat.session_id] = []
    
    chat_history[chat.session_id].append(chat.message)

    # 파일이 업로드된 경우 response_read_data 호출, 아니면 response_llama_data 호출
    if chat.session_id in upload_files:
        file_info = upload_files[chat.session_id]
        response_data = response_read_data(
            file_path=file_info["file_path"], 
            filename=file_info["filename"], 
            min_chunk_size=0
            )
    else:
        response_data = response_llama_data(prompt=chat.message)
        
    print(response_data) 
    # 스트리밍 X   
    return ChatResponse(session_id=chat.session_id, message=response_data["answer"])
    
# # 📌 answer에서 나온 답변 가져오기.
# @router.get("/answer", response_model=ChatResponse, tags=["CHAT BOT ROOT"])
# async def chat_response(chat: ChatRequest):
#     """ 데이터를 스트리밍으로 순차적으로 전송 """
#     handler = StreamHandler()
    
#     if chat.session_id in upload_files:
#         file_info = upload_files[chat.session_id]
#         response_data = response_read_data(
#             file_path=file_info["file_path"], 
#             filename=file_info["filename"], 
#             min_chunk_size=0
#         )
#     else:
#         response_data = response_llama_data(prompt=chat.message, callbacks=[handler])
            
#     handler.mark_done()

#     return StreamingResponse(handler.stream_tokens(), media_type='text/plain')
# ============================================================================== # 

@router.post("/upload", tags=["CHAT BOT ROOT"])
async def upload(session_id: int = Form(...), file: UploadFile = File(...)):
    """사용자가 업로드한 파일을 저장하고 경로를 반환"""
    try:
        # 사용자별 업로드 폴더 지정
        upload_dir = get_upload_dir(session_id)

        # 파일 저장 경로 지정
        file_location = os.path.join(upload_dir, file.filename)
        
        # 파일 저장
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 업로드된 파일 정보 저장
        upload_files[session_id] = {
            "filename": file.filename,
            "file_path": file_location
        }

        return {"session_id": session_id, "filename": file.filename, "file_path": file_location}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")

# ============================================================================== #  
# 챗봇 추가 기능 #   
# """챗봇이 예상할 수 있는 질문 리스트 반환(프론트엔드)"""
@router.get("/example", response_model=ExampleResponse, tags=["CHAT BOT ROOT"])
async def get_example_questions():
    
    example_questions = [
        "제일 싼 집 가격을 알려줘.",
        "현재 위치에서 제일 가까운 집 몇 개만 알려줘.",
        "너의 이름이 뭐고 어떤 역할을 하니?",
        "독버섯을 먹었을 떄 어떻게 해야 해?",
        "뱀에 물렸을 때 어떻게 해야 하는지 알려줘.",
        "교통안전을 지키기 위해 어떤 걸 해야 할까?"
    ]
    
    return ExampleResponse(questions=example_questions)    

# ============================================================================== # 

# 채팅 기록 표시


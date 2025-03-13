import os
import shutil
import json
import asyncio
import logging

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from fastapi import Depends, APIRouter, HTTPException, UploadFile, File, Request, Response, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from common.database import get_db
from chat_bot_root_api import response_read_data, response_llama_data
from .utils import get_upload_dir
from .objects import Query, ExampleResponse, ChatRequest, ChatResponse, ClearChatRequest
from .classified import handle_info, initial_state, handle_policy, handle_search
from langchain_community.vectorstores import FAISS

# 로깅 설정
logger = logging.getLogger(__name__)

# 이름 (지역별, 회사별 변경)
router = APIRouter(prefix="/gangjin")

chat_history = {}
upload_files = {}
# embeddings = OllamaEmbeddings(model=model_name)

# 세션 저장소: 사용자 상태를 추적
sessions: Dict[str, Dict] = {}

# UPLOAD_DIR = "./uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)  # 폴더가 없으면 생성
# ============================================================================== # 

@router.get("/activate_test", tags=["CHAT BOT ROOT"])
async def activate_test():
    """서버 활성화 테스트 엔드포인트"""
    try:
        return {"status": "active", "message": "Chat bot service is running"}
    except Exception as e:
        logger.error(f"Activation test failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error during activation test")

# ============================================================================== # 
# 채팅 세션 초기화 API
@router.get("/chat/init/{session_id}", tags=["CHAT BOT ROOT"])
async def initialize_chat(session_id: str):
    """채팅 세션 초기화 및 첫 메시지 반환"""
    # 새 세션 생성
    sessions[session_id] = {
        "state": "initial",
        "data": {}
    }
    
    # 첫 메시지 반환
    return {
        "message": "안녕하세요! 강진군 빈집 정보 챗봇입니다. 어떤 정보를 알고 싶으신가요?",
        "options": ["강진군 빈집 정보 검색", "최적의 빈집 찾기", "조건 별 검색", "관련 정책 검색"]
    }

@router.post("/chat/message", tags=["CHAT BOT ROOT"])
async def process_message(request: ChatRequest):
    """사용자 메시지 처리 및 응답 생성"""
    session_id = request.session_id
    message = request.message
    
    # 세션이 없으면 새로 생성
    if session_id not in sessions:
        sessions[session_id] = {
            "state": "initial",
            "data": {}
        }
    
    session = sessions[session_id]
    current_state = session["state"]

    # 현재 상태에 따른 메시지 처리
    if current_state == "initial":
        return initial_state(session, message)
    elif current_state == "info":
        return handle_info(session, message)
    elif current_state == "search":
        return handle_search(session, message)
    elif current_state == 'policy':
        return handle_policy(session, message)
    else:
        session["state"] = "initial"
        response_data = response_llama_data(message)  # ✅ JSON(dict)로 변환됨
        return {"message": response_data["answer"]}  # ✅ 안전하게 사용 가능
    
@router.get("/set-session", tags=["CHAT BOT ROOT"])
async def set_session(request: Request, username: str):
    request.session["user"] = {
    "username": username,
    "chat_history": []  # 대화 기록을 저장할 공간 추가
}
    return JSONResponse(content={"message": f"{username}의 대화 기록 저장 중입니다....."}, status_code=200)

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
        faiss_index_path = f"./db/faiss_index/{chat.session_id}"

        # 인덱스 존재 여부 확인
        if os.path.exists(faiss_index_path):
            # 기존 인덱스 로드
            response_data = FAISS.load_local(faiss_index_path)
        else:
            # 새로운 데이터 로드
            response_data = response_read_data(
                message=chat.message,
                file_path=file_info["file_path"], 
                filename=file_info["filename"], 
                min_chunk_size=50
            )
    # 파일이 없는 경우 일반 챗봇 응답
    else:
        response_data = response_llama_data(prompt=chat.message)

    # print(type(response_data), response_data) 
    # 스트리밍 X   
    return ChatResponse(session_id=chat.session_id, message=response_data["answer"])
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




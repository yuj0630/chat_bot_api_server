import os
import pandas as pd
import re
import unicodedata
from fastapi import HTTPException
# from common.common_utils import to_dict_data, sort_value_use_key
from dotenv import load_dotenv

load_dotenv()

session_state = []

# ============================================================================== # 
# 이전 대화기록을 출력해 주는 함수

def get_chat_history(session_id: int):
    # 세션에 해당하는 대화 내역 반환
    return session_state.get(session_id, [])

def save_message(session_id: int, message: str):
    # 세션에 메시지 추가
    if session_id not in session_state:
        session_state[session_id] = []
    session_state[session_id].append(message)

# ============================================================================== # 

# 사용자가 질문한 쿼리 전처리하는 함수
async def text_cleansing(Query: str):
    text = Query.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    if re.match(r'^[ㄱ-ㅎ]+$' or re.match(r'^[ㅏ-ㅣ]+$'), text):
        return ''
    
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    
    return text

# ============================================================================== # 
# 유니코드 정규화하는 함수
def normalize_string(s):
    return unicodedata.normalize('NFC', s)

# 사용자별 폴더 생성 함수
def get_upload_dir(session_id: int) -> str:
    """각 사용자의 session_id 기반으로 개별 폴더 생성"""
    upload_dir = os.path.join("./uploads", str(session_id), "temp")
    os.makedirs(upload_dir, exist_ok=True)  # 디렉터리가 없으면 생성
    return upload_dir
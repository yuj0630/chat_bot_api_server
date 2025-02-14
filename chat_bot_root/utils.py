import os
import io
import httpx
import pandas as pd
import re
from fastapi import HTTPException
# from common.common_utils import to_dict_data, sort_value_use_key
from dotenv import load_dotenv

from .objects import Query, ChatRequest, ChatResponse

session_state = []


# ============================================================================== # 
# 이전 대화기록을 출력해 주는 코드

def get_chat_history(session_id: str):
    # 세션에 해당하는 대화 내역 반환
    return session_state.get(session_id, [])

def save_message(session_id: str, message: dict):
    # 세션에 메시지 추가
    if session_id not in session_state:
        session_state[session_id] = []
    session_state[session_id].append(message)



# 사용자가 질문한 쿼리 전처리하는 코드

async def text_cleansing(Query: str):
    text = Query.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    if re.match(r'^[ㄱ-ㅎ]+$' or re.match(r'^[ㅏ-ㅣ]+$'), text):
        return ''
    
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    
    return text

# ============================================================================== # 
# def normalize_string(s):
#     """유니코드 정규화"""
#     return unicodedata.normalize('NFC', s)
# 
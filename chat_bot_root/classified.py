import httpx
from typing import Dict
from chat_bot_root_api import response_llama_data

API_BASE_URL = "http://localhost:8000"  # FastAPI 서버 주소 (포트 확인 필요)

# 청년지원 사업 데이터 (실제 데이터로 대체 필요)
youth_programs = [
    {
        "id": 1,
        "name": "혼인 부부 지원",
        "description": "2022. 7.4. 이후 혼인신고한 부부에게 제공되는 지원",
        "age_range": {"min": 20, "max": 100},
        "requirements": [
            "혼인신고일 기준 6개월 경과 후 1년 6개월 이내 신청",
            "부부 모두 전라남도에 6개월 이상 거주",
            "신청자는 강진군에 주소 두고 거주"
        ],
        "benefits": "1부부당 200만 원 지원",
        "application_url": "주소지 읍·면 행정복지센터 방문신청",
        "keywords": ["혼인", "부부 지원", "강진군"]
    },
    {
        "id": 2,
        "name": "결혼이민자가정 정착 지원",
        "description": "강진군에 1년 이상 주소를 두고 실거주하는 만30세 이상의 초혼 남성에게 제공되는 정착 지원",
        "age_range": {"min": 30, "max": 100},
        "requirements": [
            "신청일 기준 1년 전부터 강진군에 주소 두고 실거주",
            "초혼 남성"
        ],
        "benefits": "국제결혼에 대한 정착금 300만 원 지급",
        "application_url": "주소지 읍·면사무소 방문 신청",
        "keywords": ["결혼이민자", "정착 지원", "강진군"]
    },
    {
        "id": 3,
        "name": "임신 사전건강관리 지원",
        "description": "임신 희망 부부에게 필수 가임력검사비를 지원",
        "age_range": {"min": 15, "max": 49},
        "requirements": [
            "임신 희망 부부",
            "여성: 가임연령 (15~49세), 난소기능검사, 부인과 초음파 등",
            "남성: 정액검사"
        ],
        "benefits": "여성 최대 13만 원, 남성 최대 5만 원 지원",
        "application_url": "보건소 방문 신청 또는 정부24 온라인 신청",
        "keywords": ["임신", "건강관리", "검사비 지원"]
    },
    {
        "id": 4,
        "name": "신혼(예비)부부 건강검진비 지원",
        "description": "강진군에 주소를 두고 첫 임신을 계획 중인 예비부부 및 혼인신고 3년 이내 부부에게 건강검진비 지원",
        "age_range": {"min": 20, "max": 100},
        "requirements": [
            "첫 임신을 계획 중인 예비부부",
            "혼인신고 3년 이내 부부"
        ],
        "benefits": "검진비 본인부담금 중 여성 17만 원, 남성 9만 원 이내 지원",
        "application_url": "보건소 방문 신청",
        "keywords": ["신혼부부", "건강검진비", "강진군"]
    },
    {
        "id": 5,
        "name": "난임부부 시술비 지원",
        "description": "난임부부에게 난임 시술비를 지원",
        "age_range": {"min": 20, "max": 100},
        "requirements": [
            "난임부부"
        ],
        "benefits": "1회당 20~100만 원 차등 지원, 최대 25회 지원",
        "application_url": "보건소 방문 신청 또는 정부24 온라인 신청",
        "keywords": ["난임", "시술비 지원", "강진군"]
    },
    {
        "id": 6,
        "name": "전남형 난임부부 시술비 지원",
        "description": "전라남도에 6개월 이상 주민등록을 둔 난임부부를 위한 시술비 지원",
        "age_range": {"min": 20, "max": 100},
        "requirements": [
            "도내 6개월 이상 주민등록을 둔 난임부부",
            "정부 지원 제외자"
        ],
        "benefits": "1회당 30~150만 원 차등 지원, 횟수제한 없음",
        "application_url": "보건소 방문 신청",
        "keywords": ["난임", "시술비 지원", "전남형"]
    },
    {
        "id": 7,
        "name": "한방 난임치료 지원",
        "description": "도내 6개월 이상 주소를 둔 난임부부에게 한방치료 및 추적조사 지원",
        "age_range": {"min": 20, "max": 100},
        "requirements": [
            "도내 6개월 이상 주소를 둔 난임부부",
            "1년 이상 임신이 안된 가정"
        ],
        "benefits": "한방치료 4개월, 추적조사 3개월 지원",
        "application_url": "보건소 방문 신청",
        "keywords": ["한방 치료", "난임", "강진군"]
    }
]

# 초기 상태 처리 함수
def initial_state(session, message):
    """초기 상태에서의 사용자 메시지 처리"""
    if "강진군 빈집 정보" in message:
        session["state"] = "info"
        return {
            "message": "강진군 빈집 정보를 검색합니다. 어떤 지역의 빈집을 찾고 계신가요?",
            "options": ["강진읍", "칠량면", "대구면", "마량면", "작천면"]
        }
    elif "최적의 빈집" in message:
        session["state"] = "search"
        return {
            "message": "최적의 빈집을 찾아드리겠습니다. 원하는 집의 정보를 입력해주세요.",
            "options": ["주거용", "상업용", "농업용", "기타"]
        }
    elif "조건 별" in message:
        session["state"] = "category"
        return {
            "message": "준비 중입니다. 다른 기능을 이용해 주세요!",
            "need_confirm": True  # 확인 버튼 표시 플래그
        }
    elif "관련 정책" in message or "법" in message:
        session["state"] = "policy"
        return {
            "message": "본인에게 맞는 정책을 찾아드려요. 어떤 분야의 정책을 찾고 계신가요?",
            "options": ["교육", "취업", "결혼", "사업"]  # 확인 버튼 표시 플래그
        }
    else:
        session["state"] = "initial"
        # 인식할 수 없는 메시지 처리
        response_data = response_llama_data(prompt=message)
        
        # response_data가 문자열인지 확인
        if isinstance(response_data, str):
            # 문자열인 경우 바로 message로 사용
            return {
                "message": response_data,  # LLM에서 받은 응답 반환
                "need_confirm": False  # 확인 버튼 필요 없음
        }
        elif isinstance(response_data, dict) and "answer" in response_data:
            # response_data가 딕셔너리이고 "answer" 키가 있는 경우
            return {
                "message": response_data["answer"],  # LLM에서 받은 응답 반환
                "need_confirm": False  # 확인 버튼 필요 없음
            }
        else:
            # 예상치 못한 경우 처리 (예: 응답 형식이 잘못된 경우)
            return {
                "message": "알 수 없는 응답 형식입니다.",
                "need_confirm": False
            }

def handle_info(session: Dict, message: str) -> Dict:
    """나이 입력 처리"""
    # 숫자 추출
    age = None
    try:
        # 숫자만 추출 ("25세" -> 25)
        numbers = ''.join(filter(str.isdigit, message))
        if numbers:
            age = int(numbers)
        else:
            age = None
    except:
        age = None
    
    if age is None:
        return {
            "message": "올바른 나이를 입력해주세요. 숫자로만 입력하거나 '25세'와 같이 입력해주세요.",
            "options": ["20세", "25세", "30세", "35세", "40세"]
        }
    
    # 해당 나이에 맞는 프로그램 필터링
    matching_programs = []
    for program in youth_programs:
        if program["age_range"]["min"] <= age <= program["age_range"]["max"]:
            matching_programs.append(program)
    
    # 상태 초기화
    session["state"] = "initial"
    
    if matching_programs:
        # 챗봇 형식으로 포맷팅
        response_message = f"{age}세에 신청 가능한 지원 사업은 총 {len(matching_programs)}개입니다:\n\n"
        for i, program in enumerate(matching_programs, 1):
            response_message += f"[{i}] {program['name']}\n"
            response_message += f"- 대상: {program['age_range']['min']}세 ~ {program['age_range']['max']}세\n"
            response_message += f"- 내용: {program['description']}\n"
            if program.get("keywords"):
                response_message += f"- 키워드: {', '.join(program['keywords'])}\n"
            response_message += "\n"
        
        return {
            "message": response_message.strip(),
            "formatted": True  # 이미 포맷팅된 메시지임을 표시
        }
    else:
        return {
            "message": f"죄송합니다. {age}세에 해당하는 지원 사업을 찾지 못했습니다. 다른 나이를 입력해보시겠어요?",
            "options": ["20세", "25세", "30세", "35세", "40세"]
        }


def handle_search(session: Dict, message: str) -> Dict:
    """키워드 입력 처리"""
    keyword = message.strip().lower()
    
    if not keyword:
        return {
            "message": "검색어를 입력해주세요.",
            "options": ["취업", "창업", "주거", "금융", "교육"]
        }
    
    # 키워드 매칭 프로그램 찾기
    matching_programs = []
    for program in youth_programs:
        # 제목, 설명, 키워드에서 검색
        if (keyword in program["name"].lower() or 
            keyword in program["description"].lower() or
            any(keyword in kw.lower() for kw in program["keywords"])):
            matching_programs.append(program)
    
    # 상태 초기화
    session["state"] = "initial"
    
    if matching_programs:
        # 챗봇 형식으로 포맷팅
        response_message = f"'{keyword}' 키워드에 관련된 지원 사업은 총 {len(matching_programs)}개입니다:\n\n"
        for i, program in enumerate(matching_programs, 1):
            response_message += f"[{i}] {program['name']}\n"
            response_message += f"- 대상: {program['age_range']['min']}세 ~ {program['age_range']['max']}세\n"
            response_message += f"- 내용: {program['description']}\n"
            if program.get("keywords"):
                response_message += f"- 키워드: {', '.join(program['keywords'])}\n"
            response_message += "\n"
        
        return {
            "message": response_message.strip(),
            "formatted": True  # 이미 포맷팅된 메시지임을 표시
        }
    else:
        return {
            "message": f"죄송합니다. '{keyword}' 키워드에 관련된 지원 사업을 찾지 못했습니다. 다른 키워드로 검색해보세요.",
            "options": ["취업", "창업", "주거", "금융", "교육"]
        }
        
async def handle_policy(session: Dict, message: Dict) -> Dict:
    file_path = message.get("file_path", "")
    filename = message.get("filename", "")

    if not file_path or not filename:
        return {"status": "error", "message": "file_path 또는 filename이 누락되었습니다."}

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/gangjin/response_read_data", json={
            "file_path": file_path,
            "filename": filename
        })

    if response.status_code == 200:
        return response.json()
    else:
        return {"status": "error", "message": "response_read_data 호출 실패", "details": response.text}

# 최상위 핸들러 함수에 아래와 같은 코드 추가 필요
def process_message(session: Dict, message: str) -> Dict:
    """모든 메시지 처리의 진입점"""
    state = session.get("state", "initial")
    
    if state == "initial":
        return initial_state(session, message)
    elif state == "info":
        return handle_info(session, message)
    elif state == "search":
        return handle_search(session, message)
    elif state == "policy":
        return handle_policy(session, message)
    # elif state == "category":
    #     return handle_keyword_input(session, message)
    else:
        # 알 수 없는 상태는 초기화
        session["state"] = "initial"
        return initial_state(session, message)



# # 도구 정의
# def list_all_programs(dummy_input: str = None) -> str:
#     """모든 청년지원 사업 목록을 보여줍니다."""
#     result = "현재 제공 가능한 청년지원 사업 목록입니다:\n\n"
#     for i, program in enumerate(youth_programs, 1):
#         result += f"[{i}] {program['name']}\n"
#         result += f"- 대상: {program['age_range']['min']}세 ~ {program['age_range']['max']}세\n"
#         result += f"- 내용: {program['description']}\n"
#         result += f"- 키워드: {', '.join(program['keywords'])}\n\n"
#     return result

# def search_by_age(age_str: str) -> str:
#     """나이에 맞는 청년지원 사업을 찾습니다."""
#     try:
#         # 숫자만 추출 ("25세" -> 25)
#         numbers = ''.join(filter(str.isdigit, age_str))
#         if numbers:
#             age = int(numbers)
#         else:
#             return "올바른 나이를 입력해주세요. 
#     except:
#         return "올바른 나이를 입력해주세요.
    
#     # 해당 나이에 맞는 프로그램 필터링
#     matching_programs = []
#     for program in youth_programs:
#         if program["age_range"]["min"] <= age <= program["age_range"]["max"]:
#             matching_programs.append(program)
    
#     if matching_programs:
#         result = f"{age}세에 신청 가능한 지원 사업은 총 {len(matching_programs)}개입니다:\n\n"
#         for i, program in enumerate(matching_programs, 1):
#             result += f"[{i}] {program['name']}\n"
#             result += f"- 대상: {program['age_range']['min']}세 ~ {program['age_range']['max']}세\n"
#             result += f"- 내용: {program['description']}\n"
#             result += f"- 키워드: {', '.join(program['keywords'])}\n\n"
#         return result
#     else:
#         return f"죄송합니다. 당신에게 해당하는 지원 사업을 찾지 못했습니다."

# def search_by_keyword(keyword: str) -> str:
#     """키워드로 청년지원 사업을 검색합니다."""
#     keyword = keyword.strip().lower()
    
#     if not keyword:
#         return "검색어를 입력해주세요."
    
#     # 키워드 매칭 프로그램 찾기
#     matching_programs = []
#     for program in youth_programs:
#         # 제목, 설명, 키워드에서 검색
#         if (keyword in program["name"].lower() or 
#             keyword in program["description"].lower() or
#             any(keyword in kw.lower() for kw in program["keywords"])):
#             matching_programs.append(program)
    
#     if matching_programs:
#         result = f"'{keyword}' 로 확인된 지원 사업은 총 {len(matching_programs)}개입니다:\n\n"
#         for i, program in enumerate(matching_programs, 1):
#             result += f"[{i}] {program['name']}\n"
#             result += f"- 대상: {program['age_range']['min']}세 ~ {program['age_range']['max']}세\n"
#             result += f"- 내용: {program['description']}\n"
#             result += f"- 키워드: {', '.join(program['keywords'])}\n\n"
#         return result
#     else:
#         return f"죄송합니다. '{keyword}' 에 관련된 지원 사업을 찾지 못했습니다."

# def get_program_details(program_id: str) -> str:
#     """특정 프로그램의 상세 정보를 제공합니다."""
#     try:
#         # 프로그램 ID 처리
#         if "첫" in program_id or "처음" in program_id:
#             index = 0
#         elif "두" in program_id or "2" in program_id:
#             index = 1
#         elif "세" in program_id or "3" in program_id:
#             index = 2
#         elif "네" in program_id or "4" in program_id:
#             index = 3
#         elif "다섯" in program_id or "5" in program_id:
#             index = 4
#         else:
#             # 숫자만 추출
#             numbers = ''.join(filter(str.isdigit, program_id))
#             if numbers:
#                 index = int(numbers) - 1  # 인덱스는 0부터 시작
#             else:
#                 return "올바른 프로그램 번호를 입력해주세요. 예: '1', '첫 번째'"
#     except:
#         return "올바른 프로그램 번호를 입력해주세요."
    
#     # 프로그램 목록 체크
#     if index < 0 or index >= len(youth_programs):
#         return "죄송합니다. 해당 번호의 프로그램을 찾을 수 없습니다."
    
#     program = youth_programs[index]
    
#     # 상세 정보 제공
#     detail = f"[{program['name']} 상세 정보]\n\n"
#     detail += f"대상: {program['age_range']['min']}세 ~ {program['age_range']['max']}세\n"
#     detail += f"내용: {program['description']}\n"
#     detail += f"키워드: {', '.join(program['keywords'])}\n\n"
    
#     # 추가 정보
#     detail += f"지원 자격: {program['eligibility']}\n"
#     detail += f"신청 방법: {program['application']}\n"
#     detail += f"구비 서류: {', '.join(program['documents'])}\n"
    
#     return detail
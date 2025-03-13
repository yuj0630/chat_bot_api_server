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
            "message": "본인에게 맞는 정책을 찾아드려요. 정책 관련 문서를 업로드해주세요.",
        }
    else:
        session["state"] = "initial"
        # 인식할 수 없는 메시지 처리
        response_data = response_llama_data(prompt=message)
        print(type(response_data), response_data)
        return {"message": response_data["message"]}

def handle_info(session: Dict, message: str) -> Dict:
    """강진군 빈집 정보 안내"""
    
    keywords = ["강진군", "빈집", "주택", "공가", "임대", "매매"]
    if not any(kw in message for kw in keywords):
        return {
            "message": "빈집에 대한 정보를 원하시나요? '강진군 빈집', '빈집 매매', '공가 안내' 등의 키워드로 질문해 주세요.",
            "options": ["강진군 빈집 현황", "빈집 매매 조건", "공가 임대 신청"]
        }

    # 빈집 정보 안내 (예제)
    response_message = (
        "강진군 빈집 정보 안내입니다.\n\n"
        "- 강진군 내 공가(빈집) 현황 및 매매 정보 제공\n"
        "- 임대 및 매매 조건 안내\n"
        "- 공가 재활용 지원 정책 안내\n\n"
        "더 자세한 정보를 원하시면 '빈집 검색' 또는 '매매 가능한 빈집' 등의 키워드를 입력해 주세요."
    )

    return {
        "message": response_message,
        "options": ["빈집 검색", "매매 가능한 빈집", "임대 가능한 빈집"]
    }


def handle_search(session: Dict, message: str) -> Dict:
    """빈집 검색"""
    keyword = message.strip().lower()
    print(keyword)

    if not keyword:
        return {
            "message": "검색할 빈집 조건을 입력해주세요. (예: '강진군 빈집 매매', '임대 가능 빈집')",
            "options": ["매매 가능한 빈집", "임대 가능한 빈집", "빈집 지원 정책"]
        }

    # 빈집 데이터 (예제)
    vacant_houses = [
        {"id": 1, "location": "강진읍", "price": 5000, "status": "매매 가능"},
        {"id": 2, "location": "마량면", "price": 3000, "status": "임대 가능"},
        {"id": 3, "location": "도암면", "price": 3000, "status": "공가 지원 가능"},
        {"id": 4, "location": "군동면", "price": 4500, "status": "매매 가능"},
        {"id": 5, "location": "칠량면", "price": 2500, "status": "임대 가능"},
        {"id": 6, "location": "신전면", "price": 2500, "status": "공가 지원 가능"},
        {"id": 7, "location": "대구면", "price": 6000, "status": "매매 가능"},
        {"id": 8, "location": "성전면", "price": 3500, "status": "임대 가능"},
        {"id": 9, "location": "옴천면", "price": 5000, "status": "공가 지원 가능"},
        {"id": 10, "location": "병영면", "price": 7000, "status": "매매 가능"},
        {"id": 11, "location": "작천면", "price": 4000, "status": "임대 가능"},
        {"id": 12, "location": "강진읍", "price": 2000, "status": "공가 지원 가능"},
        {"id": 13, "location": "마량면", "price": 6000, "status": "협의 가능"},
        {"id": 14, "location": "도암면", "price": 3200, "status": "매매 가능"},
        {"id": 15, "location": "군동면", "price": 4800, "status": "임대 가능"},
    ]

    # 키워드 기반 검색 (대소문자 구분 없이, 공백 제거)
    matching_houses = [
        house for house in vacant_houses
        if keyword in house["location"].lower() or keyword in house["status"].lower()
    ]
    print(matching_houses)

    if matching_houses:
        response_message = f"'{keyword}' 조건에 맞는 빈집 목록입니다:\n\n"
        
        for i, house in enumerate(matching_houses, 1):            
            # 가격 처리 (문자열일 경우 변환)
            price = house.get("price", "0")  # 기본값 "0"
            if isinstance(price, str):
                price = price.replace("만 원", "").strip()  # "만 원" 제거
                price = int(price) if price.isdigit() else 0  # 숫자 변환, 실패 시 0
                
            response_message += f"[{i}] 위치: {house['location']}\n"
            response_message += f"- 가격: {price}만 원\n"
            response_message += f"- 상태: {house['status']}\n\n"

        return {
            "message": response_message.strip(),
            "formatted": True
        }
    else:
        return {
            "message": f"'{keyword}' 조건에 맞는 빈집을 찾을 수 없습니다. 다른 조건으로 검색해보세요.",
            "options": ["매매 가능한 빈집", "임대 가능한 빈집", "빈집 지원 정책"]
        }
        
def handle_policy(session: Dict, message: Dict) -> Dict:
    file_path = message.get("file_path", "")
    filename = message.get("filename", "")

    if not file_path or not filename:
        return {"status": "error", "message": "file_path 또는 filename이 누락되었습니다."}

    with httpx.AsyncClient() as client:
        response = client.post(f"{API_BASE_URL}/gangjin/response_read_data", json={
            "file_path": file_path,
            "filename": filename
        })

    if response.status_code == 200:
        # 비동기적으로 json 데이터를 처리합니다.
        data = response.json()
        return data
    
    else:
        return {"status": "error", "message": "문서 호출 실패", "details": response.text}

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

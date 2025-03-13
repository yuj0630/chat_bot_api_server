import json
import re

def extract_age_range(text):
    """
    연령 정보를 추출하여 {"min": X, "max": Y} 형태로 반환
    """
    age_pattern = re.findall(r"(\d{1,2})\s?[-~]\s?(\d{1,2})세", text)
    single_age_pattern = re.findall(r"만\s?(\d{1,2})세", text)
    
    if age_pattern:
        min_age, max_age = map(int, age_pattern[0])
        return {"min": min_age, "max": max_age}
    elif single_age_pattern:
        min_age = max_age = int(single_age_pattern[0])
        return {"min": min_age, "max": max_age}
    return {"min": 0, "max": 100}  # 연령 정보 없을 경우 기본값

def parse_document(text):
    """
    문서를 JSON 형태로 변환하는 함수
    """
    policies = []
    policy = {}
    
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        
        # 정책 번호 및 제목 추출
        match = re.match(r"(\d+)\s+(.+)", line)
        if match:
            if policy:
                policies.append(policy)  # 이전 정책 저장
            policy = {
                "id": int(match.group(1)), 
                "name": match.group(2), 
                "description": "", 
                "age_range": {"min": 0, "max": 100},  # 기본값 설정
                "requirements": [], 
                "benefits": "", 
                "application_url": "", 
                "contact": "",
                "keywords": []
            }
            continue
        
        # 주요 정보 추출
        if "대 상" in line:
            policy["description"] = line.replace("대 상 :", "").strip()
            policy["age_range"] = extract_age_range(line)  # 연령 정보 추출
        elif "내 용" in line:
            policy["benefits"] = line.replace("내 용 :", "").strip()
        elif "신청방법" in line:
            policy["application_url"] = line.replace("신청방법 :", "").strip()
        elif "문의처" in line or "문 의 처" in line:
            policy["contact"] = line.replace("문의처 :", "").replace("문 의 처 :", "").strip()
        elif "지원범위" in line or "조건" in line:
            policy["requirements"].append(line.strip())

    if policy:
        policies.append(policy)  # 마지막 정책 저장
    
    return policies

# 파일 불러오기
file_path = "C:\\Users\\un356\\Downloads\\PoC.txt"  # 업로드된 파일 경로
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# 문서 변환
policy_data = parse_document(text)

# JSON 저장
output_json_path = "./templates/test/policies.json"
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(policy_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON 파일이 저장되었습니다: {output_json_path}")
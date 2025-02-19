import os
import re
import httpx
import pandas as pd
from fastapi import HTTPException
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader
import pdfplumber

# 전역 함수

    
# ============================================================================== # 

# PDF 데이터 전처리 테스트

# Excel 데이터 전처리 테스트


# 한글 전처리 테스트
def preprocess_text(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = line.strip()

        # 제목 처리
        if "매뉴얼" in line or "지침" in line:
            lines.append(f"# {line}")

        # 리스트 항목 처리 (기호: ●, ○, -, o, ▶)
        elif line.startswith(("●", "○", "-", "o", "▶")):
            lines.append(f"- {line[1:].strip()}")

        # 숫자 리스트 처리 (1. 2. 3. / 1) 2) 3) / ① ② ③)
        elif re.match(r"^\d+\.\s*", line) or re.match(r"^\d+\)\s*", line) or re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", line):
            lines.append(f"- {re.sub(r'^\d+[.)]?\s*|^[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', line).strip()}")

        # 일반 텍스트
        elif line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================================== # 

# docs 내용 줄바꿈 해주는 함수
  
def format_docs(docs):
    unique_contents = set(doc.page_content for doc in docs) 
    return "\n\n".join(unique_contents)    

# 예쁘게 데이터 합치는 함수

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

# ============================================================================== # 

# 파일 전처리 함수(한글 태아불 -> 마크다운)
def convert_hwp_to_markdown(hwp_table_text: str):
    soup = BeautifulSoup(hwp_table_text, 'html.parser')
    table = soup.find('table')
    df = pd.read_html(str(table))[0]
    markdown_table = df.to_markdown(index=False)
    return markdown_table
    
    
    
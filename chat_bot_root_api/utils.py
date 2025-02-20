import re
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader
import pdfplumber
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, PDFPlumberLoader, DataFrameLoader, DirectoryLoader, TextLoader
from langchain_teddynote.document_loaders import HWPLoader

# 전역 함수



# ================      파일 다운로드 부분       ========================= #     
# 확장자별 파일 다운로드 함수 (현재는 load문 사용, 필요시 text로 변경)
def load_file(file_path: str, filename: str):
    file_type = filename.split('.')[-1].lower()

    if file_type == "pdf":
        loader = PDFPlumberLoader(file_path)
        pages = loader.load()
        
    elif file_type == "xlsx" or file_type == 'xls': 
        df = pd.read_excel(file_path)
        df = df.dropna(subset=["answer"])
        loader = DataFrameLoader(df, page_content_column="answer")
        pages = loader.load()
        
    elif file_type == "csv":
        df = pd.read_csv(file_path, encoding="EUC-KR")
        df = df.dropna(subset=["긴급구조분류명"])
        loader = DataFrameLoader(df, page_content_column="긴급구조분류명")
        pages = loader.load()
        
    elif file_type == 'hwp':
        loader = HWPLoader(file_path)
        pages = loader.load()
        
        # text = "\n".join(page.page_content for page in pages)
        # preprocess_text_origin = preprocess_text(text)
        # print('=====================')
        # print(preprocess_text_origin)
    else:
        raise ValueError(f"지원되지 않는 파일 형식: {file_type}")
    
    
    print(pages)
    return pages

# ==================      일반 함수 부분      ============================= # 

# docs 내용 줄바꿈 해주는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
# =================          전처리 부분        ============================ # 

# 파일 전처리 함수(한글 테이블 -> 마크다운)
def convert_hwp_to_markdown(hwp_table_text: str):
    soup = BeautifulSoup(hwp_table_text, 'html.parser')
    table = soup.find('table')
    df = pd.read_html(str(table))[0]
    markdown_table = df.to_markdown(index=False)
    return markdown_table

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
    
# ====================       성능 향상 부분       ======================== # 

# # 벡터화 처리 배치화(데이터 다운로드 batch해서 빠르게 해줌)
# def batch_vectorize(text_chunks, embeddings, vector_store, batch_size=10):
#     """텍스트 덩어리를 배치로 벡터화하여 저장하는 함수"""
#     for i in tqdm(range(0, len(text_chunks), batch_size), total=len(text_chunks)//batch_size):
#         batch = text_chunks[i:i+batch_size]
#         vector_store.add_documents(batch)  # 배치로 벡터화하여 저장
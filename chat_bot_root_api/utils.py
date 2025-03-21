import re
import pandas as pd
from tqdm import tqdm
import io
import base64
import pypdfium2

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, PyPDFium2Loader, DataFrameLoader, DirectoryLoader, TextLoader
from langchain_teddynote.document_loaders import HWPLoader
import pytesseract
# 전역 함수



# ================      파일 다운로드 부분       ========================= #     
# 확장자별 파일 다운로드 함수 (현재는 load문 사용, 필요시 text로 변경)
def load_file(file_path: str, filename: str):
    file_type = filename.split('.')[-1].lower()

    if file_type == "pdf":
        loader = PyPDFium2Loader(file_path)
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

# ====================       성능 향상 부분       ======================== # 

# PDF page에서 텍스트 추출하는 함수 작성
def extract_text_with_ocr(page):
    text = page.extract_text()
    if not text: # 만약 추출할 텍스트가 없다면
        # PDF page를 이미지로 변환
        image = page.to_image()
        # 이미지에서 OCR 재실행하여 텍스트 추출
        text = pytesseract.image_to_string(image)
    return text

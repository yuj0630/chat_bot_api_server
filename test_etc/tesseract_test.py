import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Tesseract 경로 설정 (Windows의 경우, 설치 후 경로를 지정해야 할 수 있음)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows 경로 예시

# PDF 파일에서 이미지를 추출하고 OCR로 텍스트 추출
def extract_text_from_pdf(pdf_path: str) -> str:
    # PDF에서 페이지를 이미지로 변환
    poppler_path = 'C:\\poppler-24.08.0\\Library\\bin'
    
    # PDF에서 페이지를 이미지로 변환
    pages = convert_from_path(pdf_path, 300, poppler_path=poppler_path)
    
    extracted_text = ""
    
    # 각 페이지의 이미지를 OCR로 텍스트 변환 (한국어 인식 추가)
    for page in pages:
        # 이미지에서 텍스트 추출 (한국어 추가)
        page_text = pytesseract.image_to_string(page, lang='kor')  # 한국어 언어 설정
        extracted_text += page_text + "\n"
    
    return extracted_text

# 사용 예시
pdf_path = "C:/Users/un356/Downloads/population_guidebook_2408.pdf"
text = extract_text_from_pdf(pdf_path)
print(text)
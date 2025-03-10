from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# PDF에서 이미지를 추출하고 OCR로 텍스트 추출
def extract_text_from_pdf(pdf_path: str) -> str:
    # PDF에서 페이지를 이미지로 변환
    poppler_path = 'C:\\poppler-24.08.0\\Library\\bin'  # Poppler 경로 설정
    pages = convert_from_path(pdf_path, 300, poppler_path=poppler_path)
    
    # PaddleOCR 초기화 (영어와 한국어 모델 사용)
    ocr = PaddleOCR(use_angle_cls=True, lang='korean')  # 한국어 OCR 모델 사용
    
    extracted_text = ""
    
    for page in pages:
        # PIL.Image 객체를 numpy 배열로 변환
        page_np = np.array(page)
        
        # 페이지 이미지를 OCR 처리
        page_text = ocr.ocr(page_np, cls=True)
        
        # OCR 결과에서 텍스트만 추출
        for result in page_text[0]:
            text = result[1][0]
            extracted_text += text + "\n"
    
    return extracted_text

# 사용 예시
pdf_path = "C:/Users/un356/Downloads/population_guidebook_2408.pdf"
text = extract_text_from_pdf(pdf_path)
print(text)
import os
import re
import nltk
import multiprocessing
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class PDFRAGProcessor:
    def __init__(
        self,
        min_words_per_page: int = 30,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 100,
        num_workers: int = 4,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./faiss_db"
    ):
        """
        PDF RAG 시스템 초기화
        
        Args:
            min_words_per_page: 페이지당 최소 단어 수 (이 값 미만인 페이지는 제외)
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩 크기
            batch_size: 배치 처리 크기
            num_workers: 병렬 처리에 사용할 워커 수
            embedding_model: 임베딩 모델 이름
            persist_directory: 벡터 저장소 위치
        """
        self.min_words_per_page = min_words_per_page
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.num_workers = min(num_workers, multiprocessing.cpu_count())
        
        # 텍스트 스플리터 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # 개인정보 분석 및 익명화 엔진 초기화
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # 벡터 저장소 초기화
        self.persist_directory = persist_directory
        self.vectorstore = None

    def _load_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """PDF 파일 로드 및 페이지별 메타데이터 추가"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # 파일 이름 추출
        filename = os.path.basename(pdf_path)
        
        # 메타데이터 추가
        for i, page in enumerate(pages):
            page.metadata["filename"] = filename
            page.metadata["page_number"] = i + 1
            
        return pages
    
    def _filter_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """페이지 필터링 (단어 수가 최소 기준 미만인 페이지 제외)"""
        filtered_pages = []
        
        for page in pages:
            # 단어 수 계산
            word_count = len(page.page_content.split())
            
            # 메타데이터에 단어 수 추가
            page.metadata["word_count"] = word_count
            
            # 최소 단어 수 기준 충족 여부 확인
            if word_count >= self.min_words_per_page:
                filtered_pages.append(page)
                
        return filtered_pages
    
    def _process_tables(self, text: str) -> str:
        """표 형식 텍스트 처리 (표 감지 및 구조화)"""
        # 표 패턴 감지 (간단한 규칙 기반)
        table_pattern = r"(\+[-+]+\+\n(\|[^|]+\|[^|]+\|\n)+\+[-+]+\+)"
        
        def format_table(match):
            table_text = match.group(0)
            # 여기서 표 형식을 유지하면서 특별한 처리 가능
            # 예: markdown 표 형식으로 변환 또는 구조화된 형태로 보존
            return f"\n<TABLE>\n{table_text}\n</TABLE>\n"
        
        # 표 패턴 처리
        processed_text = re.sub(table_pattern, format_table, text)
        return processed_text
    
    def _remove_pii(self, text: str) -> str:
        """개인정보(PII) 제거"""
        # 분석할 엔티티 유형 정의
        pii_entities = [
            "PERSON", "PHONE_NUMBER", "ID_NUMBER", "CREDIT_CARD", "EMAIL_ADDRESS"
        ]
        
        # 텍스트에서 개인정보 감지
        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=pii_entities,
            language="ko"  # 한국어 설정 (변경 가능)
        )
        
        # 감지된 개인정보 익명화
        anonymized_text = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results
        ).text
        
        return anonymized_text
    
    def _preprocess_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """문서 전처리 (표 처리 및 개인정보 제거)"""
        # 표 형식 처리
        processed_text = self._process_tables(document.page_content)
        
        # 개인정보 제거
        anonymized_text = self._remove_pii(processed_text)
        
        # 처리된 텍스트로 업데이트
        document.page_content = anonymized_text
        return document
    
    def _process_batch(self, batch_pdfs: List[str]) -> List[Dict[str, Any]]:
        """PDF 배치 처리"""
        all_documents = []
        
        for pdf_path in batch_pdfs:
            try:
                # PDF 로드
                pages = self._load_pdf(pdf_path)
                
                # 페이지 필터링
                filtered_pages = self._filter_pages(pages)
                
                # 각 페이지 전처리
                processed_pages = [self._preprocess_document(page) for page in filtered_pages]
                
                all_documents.extend(processed_pages)
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
        
        return all_documents
    
    def process_pdfs(self, pdf_paths: List[str]) -> None:
        """PDF 파일 목록 처리 및 분산 처리"""
        # 작업 배치로 나누기
        batches = [pdf_paths[i:i + self.batch_size] for i in range(0, len(pdf_paths), self.batch_size)]
        
        all_documents = []
        
        # 단일 프로세스 처리 (디버깅용) 또는 멀티프로세싱
        if self.num_workers <= 1:
            for batch in tqdm(batches, desc="Processing PDF batches"):
                documents = self._process_batch(batch)
                all_documents.extend(documents)
        else:
            # 멀티프로세싱 풀 생성
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                # 배치 병렬 처리
                results = list(tqdm(
                    pool.imap(self._process_batch, batches),
                    total=len(batches),
                    desc="Processing PDF batches"
                ))
                
                # 결과 취합
                for documents in results:
                    all_documents.extend(documents)
        
        print(f"Total documents after filtering: {len(all_documents)}")
        
        # 문서 분할
        texts = self.text_splitter.split_documents(all_documents)
        print(f"Total chunks after splitting: {len(texts)}")
        
        # 벡터 저장소에 저장
        self.vectorstore = Chroma.from_documents(
            texts, 
            self.embeddings, 
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        
        print(f"Vectorstore created with {len(texts)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """쿼리 기반 검색"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Process documents first.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        return results
    
    def analyze_document_stats(self) -> pd.DataFrame:
        """처리된 문서 통계 분석"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Process documents first.")
        
        # Chroma에서 모든 문서와 메타데이터 가져오기
        documents = self.vectorstore.get()
        metadatas = documents['metadatas']
        
        # DataFrame으로 변환
        df = pd.DataFrame(metadatas)
        
        # 기본 통계
        stats = {
            "total_documents": len(df),
            "unique_files": df['filename'].nunique() if 'filename' in df.columns else 0,
            "avg_word_count": df['word_count'].mean() if 'word_count' in df.columns else 0,
        }
        
        print("Document Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        return df

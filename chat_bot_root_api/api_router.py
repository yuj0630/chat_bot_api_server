import os
from tqdm import tqdm
from fastapi import APIRouter
import pandas as pd
from .model import setup_llm_pipeline
from .utils import preprocess_text

# langchain 모듈
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, PDFPlumberLoader, DataFrameLoader, DirectoryLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_teddynote.document_loaders import HWPLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser

# 파인튜닝 
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.retrievers import BM25Retriever 
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import LongContextReorder
from sentence_transformers import CrossEncoder


router = APIRouter(prefix="/api/chat_bot_root") # APIRouter 변환

model_name = "llama3.2-bllossom" 

llm = ChatOllama(model=model_name) 

print(f"사용되는 모델: {llm}")  # 모델 이름 출력

# 입력된 PDF 없을 시 Llama 모델을 사용하여 기본적인 응답 생성하는 코드
@router.get("/response_llama_data",  tags=["CHAT BOT API SERVER"])
def response_llama_data(prompt : str):
    # 모델에 메시지 전달
    try:
        # System Prompt 적용 (위에서 설계한 버전)
        system_prompt = """당신은 재난 안전관리 전문가로, 사용자의 질문에 대해 정확하고 공손하게 답변해야 합니다. 
        PDF 및 TXT 데이터를 입력받으면 해당 데이터의 요약 또는 사용자가 원하는 정보를 제공합니다.

        ### 🔹 **📌 핵심 원칙**
        1. **재난 안전관리 관련 질문**: 
            - 신뢰할 수 있는 정보를 바탕으로 재난 대응 및 예방 지침을 제공합니다.
            - 체계적인 단계별 설명(CoT, Chain of Thought 방식)을 포함하여 논리적이고 명확한 답변을 생성합니다.
    
        2. **파일(PDF, TXT) 입력 시**:
            - 사용자가 원하면 **파일 요약, 특정 내용 검색 및 정리**를 수행합니다.
            - 문서 내용을 정확하게 분석하여 필요한 정보를 추출합니다.

        3. **언어 정책**:
            - 기본적으로 **모든 답변은 한국어**로 작성됩니다.
            - 질문에 **한국어가 포함된 경우** 최대한 한국어로 답변합니다.
            - 질문이 **한국어가 아닌 경우**, 해당 언어로 답변할 수 있습니다.

        4. **답변 스타일**:
            - **공손하고 정중한 어조**로 답변합니다.
        """

        # Ollama에서 요구하는 메시지 형식으로 변환
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = llm.invoke(messages)
        
        # ✅ AIMessage 객체에서 content(문자열) 값만 가져오기
        answer_text = response.content if hasattr(response, 'content') else str(response)
        
        answer = {"answer" : answer_text} #JSON 형식으로 리턴
        
        return answer  # 응답 반환
    
    except Exception as e:
        return f"오류 발생: {str(e)}"
        
# ========================================================================= # 
# 입력된 데이터 있을 시 해당 데이터 읽고 학습하는 코드     
@router.get("/response_read_data",  tags=["CHAT BOT API SERVER"]) 
def response_read_data(file_path: str, filename: str, min_chunk_size : int):
    """데이터 파일을 읽고, 벡터화하는 함수"""
    # 모델 초기화
    try:
        # 모델 이름 로깅
        print(f"사용되는 모델: {llm}")  # 모델 이름 출력
        
        file_type = filename.split('.')[-1].lower()
        print(file_type)
        
        target_path = f'./templates/{file_path}/{filename}'
        
        print(target_path)
        
        # 자료형별로 파일 로드
        if file_type == "pdf":
            loader = PDFPlumberLoader(target_path)
            pages = loader.load()
            
        elif file_type == "xlsx" or file_type == 'xls': 
            df = pd.read_excel(target_path)
            df = df.dropna(subset=["answer"])
            loader = DataFrameLoader(df, page_content_column="answer") # 여기 주기적으로 바꿔줘야 함
            pages = loader.load()
            
        elif file_type == "csv":
            df = pd.read_csv(target_path, encoding="EUC-KR")  # 인코딩 명확히 지정
            df = df.dropna(subset=["긴급구조분류명"]) 
            loader = DataFrameLoader(df, page_content_column="긴급구조분류명")
            pages = loader.load()
            
        elif file_type == 'hwp': #hwp5txt
            loader = HWPLoader(target_path)
            pages = loader.load()
            
            # pages가 리스트 형태라면 개별 요소를 합쳐서 문자열로 변환
            text = "\n".join(page.page_content for page in pages)
            
            # 전처리 적용
            preprocess_text_origin = preprocess_text(text)
            print('=====================')
            print(preprocess_text_origin)
            
        #  format_doc(pages)
        print(pages)
            
        if not pages:
            raise ValueError("파일에서 텍스트를 추출할 수 없습니다.")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        # 텍스트 청킹
        text_chunks = text_splitter.split_documents(pages)
        
        
        # 3. 최소 크기 필터링
        filter_text_chunks = [
            doc for doc in text_chunks if len(doc.page_content.strip()) >= int(min_chunk_size)
        ]
        
        
        # Embedding과 Vector Store 설정
        embeddings = OllamaEmbeddings(model=model_name)  # 사용하려는 Embedding 모델
        
        vector_store = Chroma.from_documents(filter_text_chunks, embeddings)
        
        for i, chunk in tqdm(enumerate(filter_text_chunks), total=len(filter_text_chunks), desc="Vectorizing documents"):
            # 각 텍스트 덩어리를 벡터화하여 저장
            vector_store.add_documents([chunk])
        
        print(type(vector_store))
        # 이제 벡터스토어 중 chroma, FAISS, bm25, finecone(유료), pgvector 중 하나 선택
        
        # # Retriever 설정(chroma, FAISS, bm25)
        # chroma_retriever = vector_store.as_retriever(
        #     search_type="similarity",
        #     search_kwargs={"k": 3}
        #     )
        
        FAISS_vectorstore = FAISS.from_documents(documents=filter_text_chunks,
                                                embedding=embeddings,
                                                distance_strategy = DistanceStrategy.COSINE,
                                                )
        faiss_retriever = FAISS_vectorstore.as_retriever()
        
        bm25_retriever = BM25Retriever.from_documents(filter_text_chunks)
        bm25_retriever.k = 5
        
        
        # 앙상블 retriever(2개 이상)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )
        
        # # 리랭커 (어울리는 순위 재조정 실험 중)
        # model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        # compressor = CrossEncoderReranker(model=model, top_n=3)
        
        # compression_retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor, base_retriever=ensemble_retriever)
        
        
        # Retriever 타입 확인
        # print(f"Chroma Retriever 타입: {type(chroma_retriever)}")
        print(f"FAISS Retriever 타입: {type(faiss_retriever)}")
        print(f"BM25Retriver 타입: {type(bm25_retriever)}")
        print(f"Ensemble Retriever 타입: {type(ensemble_retriever)}")
        # print(f"compression Retriever 타입: {type(compression_retriever)}")
        
        # 템플릿 설정
        prompt = ChatPromptTemplate.from_messages(
            [
                # System Prompt 적용 (위에서 설계한 버전)
                ("system", """당신은 재난관리 전문가로, 사용자의 질문에 대해 정확하고 공손하게 답변해야 합니다. 
                파일 데이터를 입력받고 해당 데이터의 질문에 맞게 사용자가 원하는 정보를 제공합니다.
                데이터의 내용은 간략하게 설명해야 합니다.
                
                ### 🔹 **📌 핵심 원칙**
                
                1. **재난안전 관련 질문**: 
                    - 입력받은 데이터 내에 있는 정보를 바탕으로 고객이 원하는 정보의 답을 생성합니다. 
                    - 체계적인 단계별 설명(CoT, Chain of Thought 방식)을 포함하여 논리적이고 명확한 답변을 생성합니다.
                    
                2. **파일 입력 시**:
                    - 문서 내용을 분석한 후 질문의 내용에 기반하여 필요한 정보를 추출합니다.
                    - 사용자가 질문에 요약 및 검색을 원하면 **파일 요약, 특정 내용 검색 및 정리**를 수행합니다.
                    - 동일한 내용을 반복하지 않습니다.
                    
                3. **언어 정책**:
                    - 기본적으로 **모든 답변은 한국어**로 작성됩니다.
                    - 질문에 **한국어가 포함된 경우** 최대한 한국어로 답변합니다.
                    - 질문이 **한국어가 아닌 경우**, 해당 언어로 답변할 수 있습니다.
                
                4. **답변 스타일**:
                    - **공손하고 정중한 어조**로 답변합니다.
                    - 마지막에 **추가 질문**을 요구합니다.
                """),
                ("human", "질문: {question}")
            ]
        )
        
        # Chain 생성
        chain = ensemble_retriever | prompt | llm | StrOutputParser() 
        
        response = chain.invoke("낙뢰가 일어난 뒤에 정전이 일어나면 어떻게 해야 할까?") 
        
        print(response)
        
        answer = {"answer" : response} #JSON 형식으로 리턴
        
        return answer  # 생성된 QA 체인 반환
    
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        raise

# ========================================================================= # 
# AGENTIC RAG 설정하는 코드

# @router.get("/response_agent",  tags=["CHAT BOT API SERVER"])
# def response_agent(prompt : str):
#     """주제별 agent 템플릿 설정하는 함수"""
    
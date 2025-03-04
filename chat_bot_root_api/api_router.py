import os
from tqdm import tqdm
from fastapi import APIRouter, HTTPException
import pandas as pd
import logging
from .model import setup_llm_pipeline
from .utils import load_file
from transformers import pipeline

# langchain 모듈
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, PDFPlumberLoader, DataFrameLoader, DirectoryLoader, TextLoader
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

# 에이전트
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, create_json_chat_agent
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 로깅 설정
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/api/chat_bot_root") # APIRouter 변환

upload_files = {}
model_name = "llama3.2-bllossom" 


llm = setup_llm_pipeline()

# llm = ChatOllama(model=model_name) 
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
        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": prompt}
        # ]
        
        # 응답 처리 (텍스트 추출)
        # answer_text = response[0]['generated_text'] if isinstance(response, list) else str(response)
        
        messages = [
            {"role": "user", "content": "Who are you?"},
            ]
        
        pipe = pipeline("text-generation", model="BAAI/bge-reranker-v2-m3", trust_remote_code=True)
        pipe(messages)
        
        
        
        answer = {"answer" : messages} #JSON 형식으로 리턴
        print(answer)
        
        return answer  # 응답 반환
    
    except Exception as e:
        return f"오류 발생: {str(e)}"
        
# ========================================================================= # 
# 입력된 데이터 있을 시 해당 데이터 읽고 학습하는 코드     
@router.get("/response_read_data",  tags=["CHAT BOT API SERVER"]) 
def response_read_data(session_id : int, file_path: str, filename: str, min_chunk_size : int):
    """데이터 파일을 읽고, 벡터화하는 함수"""
    # 모델 초기화
    try:
        # # 세션에서 파일 정보 읽기
        # file_info = upload_files.get(session_id)
        # if not file_info:
        #     raise HTTPException(status_code=400, detail="파일 정보가 존재하지 않습니다.")
        
        # # 파일 경로와 이름
        # file_path = file_info["file_path"]
        # filename = file_info["filename"]
        
        file_type = filename.split('.')[-1].lower()
        target_path = f'./templates/{file_path}/{filename}'
        
        # 함수화 중
        # load_file(file_path, filename)
        
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
            
        #  format_doc(pages)
        print(pages)
            
        if not pages:
            raise ValueError("파일에서 텍스트를 추출할 수 없습니다.")
        
        # 텍스트 분할 및 청킹
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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
        print(f"FAISS Retriever 타입: {type(faiss_retriever)}")
        print(f"BM25Retriver 타입: {type(bm25_retriever)}")
        print(f"Ensemble Retriever 타입: {type(ensemble_retriever)}")
        
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
        
        answer = {"answer" : response} # JSON 형식으로 리턴
        return answer  # 생성된 QA 체인 반환
    
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        raise

# ========================================================================= # 
# AGENTIC RAG 설정하는 코드(ex.강진)

@router.get("/response_agent",  tags=["CHAT BOT API SERVER"])
def response_agent(session_id: int, prompt : str):
    """주제별 agent 템플릿 설정하는 함수"""
    
    ### PDF 문서 검색 도구 (Retriever) ###
    # 여기서 질문에 맞는 데이터를 다운로드하며, DB가 될 수도 있습니다.
    loader = TextLoader(prompt)

    # 텍스트 분할기를 사용하여 문서를 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # 문서를 로드하고 분할합니다.
    split_docs = loader.load_and_split(text_splitter)

    # VectorStore를 생성합니다.
    vector = FAISS.from_documents(split_docs, OllamaEmbeddings())

    # Retriever를 생성합니다.
    retriever = vector.as_retriever()
    
    
    # 검색기 및 tool 다양화입니다. 실전에서는 집 내용, 집 가격, 강진 내 정보 등 특정 분야에서 다양화할거예요.
    retriever_tool = create_retriever_tool(
    retriever,
    "db_search",
    "DB에서 찾은 정보들입니다. DB 내 정보를 찾고 싶을 때 해당 검색기를 주로 이용하세요!",
    )
    
    retriever_tool2 = create_retriever_tool(
    retriever,
    "web_search",
    "웹 검색을 통해 찾은 정보들입니다. 웹사이트 내 정보를 찾고 싶을 때 해당 검색기를 주로 이용하세요!",
    )
    
    retriever_tool3 = create_retriever_tool(
    retriever,
    "RAG_search",
    "RAG 검색을 통해 찾은 정보들입니다. RAG 데이터 내 정보를 찾고 싶을 때 해당 검색기를 주로 이용하세요!",
    )

    # 해당 agent 뒤 쪽에 json_prompt일 시 react -> json_chat으로 변경
    agent = create_react_agent(llm, retriever_tool)
    agent2 = create_react_agent(llm, retriever_tool2)
    agent3 = create_react_agent(llm, retriever_tool3)
    
    # agent_executor의 경우 tool 및 agent를 List화 하여 상황에 맞는 retriever를 사용합니다.
    agent_executor = AgentExecutor(
        agent=agent,  # agent 선택
        tools=retriever_tool, # 검색기 툴 선택
        verbose=True, # 언어 여부
        handle_parsing_errors=True, # parser 에러시 임의 복구
        return_intermediate_steps=True, # 중간과정
    )
    
    # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # 대부분의 실제 시나리오에서 세션 ID가 필요하기 때문에 이것이 필요합니다
        # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
        lambda session_id: session_id,
        # 프롬프트의 질문이 입력되는 key: "input"
        input_messages_key="input",
        # 프롬프트의 메시지가 입력되는 key: "chat_history"
        history_messages_key="chat_history",
        )
    
    # 질의에 대한 답변을 출력합니다.
    response = agent_with_chat_history.invoke(
        {
            "input": "강진 빈집에 대한 질문 내용을 DB 내 정보에서 알려줘"
        },
        # 세션 ID를 설정합니다.
        config={"configurable": {"session_id": session_id}},
    )
    print(f"답변: {response['output']}")

# 해당 부분은 agent 실험용으로, 실전에선 안 쓰는 코드예요.
# ========================================================================= # 

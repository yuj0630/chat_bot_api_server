import os
import io
import base64
from tqdm import tqdm
from fastapi import APIRouter, HTTPException
import pandas as pd
import logging
from .model import setup_llm_pipeline
from .utils import load_file
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

# langchain 모듈
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, PDFPlumberLoader, DataFrameLoader, Docx2txtLoader, TextLoader, JSONLoader
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

# 병렬처리 (학습속도 가속)
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 로깅 설정
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/gangjin") # APIRouter 변환

upload_files = {}
model_name = "llama3.2-bllossom" 


# llm = setup_llm_pipeline()

llm = ChatOllama(model=model_name) 
print(f"사용되는 모델: {llm}")  # 모델 이름 출력

@router.get("/response_llama_data",  tags=["CHAT BOT API SERVER"]) 
def response_llama_data(prompt : str):
    # 모델에 메시지 전달
    try:
        # System Prompt 적용 (위에서 설계한 버전)
        system_prompt = """당신은 강진에서 제공하는 챗봇 안내원으로, 사용자의 질문에 대해 정확하고 공손하게 답변해야 합니다. 
                파일 데이터가 있으면 해당 데이터를 정확히 읽고 필요한 정보를 간략하게 제공해야 합니다.

                ### 🔹 **📌 핵심 원칙**
                1. **답변 시 유의 사항**: 
                    - 입력받은 데이터 내에 있는 정보를 바탕으로 고객이 질문하는 정보의 답을 생성합니다.
                    - 추가 정보가 필요할 시 해당 정보를 다시 물어봐야 합니다.
            
                2. **언어 정책**:
                    - 기본적으로 **모든 답변은 한국어**로 작성됩니다.
                    - 질문에 **한국어가 포함된 경우** 최대한 한국어로 답변합니다.
                    - 질문이 **한국어가 아닌 경우**, 해당 언어로 답변할 수 있습니다.
                
                3. **답변 스타일**:
                    - 답변의 끝에는 추가로 필요한 정보를 제공할 수 있도록 **친절한 안내 문구**를 포함합니다.
                    - **중복을 피하고, 핵심 정보만 간결하고 정확하게 제공**합니다.
                """

        # Ollama에서 요구하는 메시지 형식으로 변환
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = llm.invoke(messages)
        
        # ✅ AIMessage 객체에서 content(문자열) 값만 가져오기
        answer_text = response.content if hasattr(response, 'content') else str(response)
        print(type(answer_text), answer_text[:500])
        
        # 이제 response_data["answer"]를 사용할 수 있습니다
        return {"message": answer_text[:500]}
    
    except Exception as e:
        return f"오류 발생: {str(e)}"
        
# ========================================================================= # 
# 입력된 데이터 있을 시 해당 데이터 읽고 학습하는 코드     
@router.get("/response_read_data",  tags=["CHAT BOT API SERVER"]) 
def response_read_data(message: str, file_path: str, filename: str, min_chunk_size : int = 50):
    """데이터 파일을 읽고, 벡터화하는 함수"""
    # 모델 초기화
    try:
        # 파일 경로와 이름
        target_path = file_path
        file_type = filename.split('.')[-1].lower()
        
        # file_type = filename.split('.')[-1].lower()
        # target_path = f'./templates/{file_path}/{filename}'
        
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
            
        elif file_type == 'hwp' or file_type == 'hwpx': #hwp5txt
            loader = HWPLoader(target_path)
            pages = loader.load()
            
        elif file_type == 'txt':
            loader = TextLoader(target_path, encoding='utf-8')  # 인코딩을 utf-8로 지정
            pages = loader.load()
            
        elif file_type == 'json':
            loader = JSONLoader(
                file_path = target_path,
                text_content=False
            )
        #  format_doc(pages)
        print(pages)
            
        if not pages:
            raise ValueError("파일에서 텍스트를 추출할 수 없습니다.")
        
        # 텍스트 분할 및 청킹
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(pages)
        
        # 3. 최소 크기 필터링
        filter_text_chunks = [
            doc for doc in text_chunks if len(doc.page_content.strip()) >= int(min_chunk_size)
        ]
        
        # Embedding과 Vector Store 설정
        embeddings = OllamaEmbeddings(model=model_name)  # 사용하려는 Embedding 모델
        
        ## Retriever 설정(chroma, FAISS, bm25)
        ## Chroma retriever 사용해서 학습
        # vector_store = Chroma.from_documents(filter_text_chunks, embeddings) 
        
        # for i, chunk in tqdm(enumerate(filter_text_chunks), total=len(filter_text_chunks), desc="Vectorizing documents"):
        #     # 각 텍스트 덩어리를 벡터화하여 저장
        #     vector_store.add_documents([chunk])
        
        # chroma_retriever = vector_store.as_retriever(
        #     search_type="similarity",
        #     search_kwargs={"k": 3}
        #     )
    
        # FAISS_vectorstore 이용해서 학습
        FAISS_vectorstore = FAISS.from_documents(documents=filter_text_chunks,
                                                embedding=embeddings,
                                                distance_strategy = DistanceStrategy.COSINE,
                                                )
        FAISS_vectorstore.save_local("./db/faiss_index/{session_id}")
        faiss_retriever = FAISS_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        
        bm25_retriever = BM25Retriever.from_documents(filter_text_chunks)
        bm25_retriever.k = 3
        
        
        # 앙상블 retriever(2개 이상)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        
        query = message
        # FAISS와 BM25에서 검색된 문서들
        faiss_docs = faiss_retriever.get_relevant_documents(query)
        bm25_docs = bm25_retriever.get_relevant_documents(query)
        ensemble_docs = ensemble_retriever.get_relevant_documents(query)
        
        # FAISS에서 검색된 문서 표시
        print("### FAISS 검색된 문서 ###")
        for rank, doc in enumerate(faiss_docs, start=0):
            print(f"Rank: {rank}")
            print(f"Document Title: {doc.metadata.get('title', 'No Title')}")
            print(f"Document Content: {doc.page_content}\n")

        # BM25에서 검색된 문서 표시
        print("### BM25 검색된 문서 ###")
        for rank, doc in enumerate(bm25_docs, start=0):
            print(f"Rank: {rank}")
            print(f"Document Title: {doc.metadata.get('title', 'No Title')}")
            print(f"Document Content: {doc.page_content}\n")
            
        # BM25에서 검색된 문서 표시
        print("### 앙상블 검색된 문서 ###")
        for rank, doc in enumerate(ensemble_docs, start=0):
            print(f"Rank: {rank}")
            print(f"Document Title: {doc.metadata.get('title', 'No Title')}")
            print(f"Document Content: {doc.page_content}\n")
        
        
        # # ✅ Rerank 모델 로드
        # reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # # ✅ 2. RAG에서 검색된 문서 가져오기
        # query = "사용자의 질문"  # 🔵 실제 사용자 입력값으로 대체
        # retrieved_docs = ensemble_retriever.get_relevant_documents(query)

        # # ✅ 3. Rerank 적용 (유사도 점수 기반 정렬)
        # document_texts = [doc.page_content for doc in retrieved_docs]  # 🔵 검색된 문서 리스트
        # scores = reranker.predict([(query, doc) for doc in document_texts])  # 🔵 문장-질문 유사도 평가
        # reranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)]

        # # ✅ 4. 최종 상위 문서 선택 (Top 5)
        # final_docs = reranked_docs[:5]
        
        # 템플릿 설정
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """당신은 강진에서 제공하는 챗봇 안내원으로, 사용자의 질문에 대해 정확하고 공손하게 답변해야 합니다. 
                파일 데이터가 있으면 해당 데이터를 정확히 읽고 필요한 정보를 간략하게 제공해야 합니다.

                ### 🔹 **📌 핵심 원칙**
                1. **답변 시 유의 사항**: 
                    - 입력받은 데이터 내에 있는 정보를 바탕으로 고객이 질문하는 정보의 답을 생성합니다.
                    - 추가 정보가 필요할 시 해당 정보를 다시 물어봐야 합니다.
            
                2. **언어 정책**:
                    - 기본적으로 **모든 답변은 한국어**로 작성됩니다.
                    - 질문에 **한국어가 포함된 경우** 최대한 한국어로 답변합니다.
                    - 질문이 **한국어가 아닌 경우**, 해당 언어로 답변할 수 있습니다.
                
                3. **답변 스타일**:
                    - 답변의 끝에는 추가로 필요한 정보를 제공할 수 있도록 **친절한 안내 문구**를 포함합니다.
                    - **중복을 피하고, 핵심 정보만 간결하고 정확하게 제공**합니다.
                """),
                ("human", "{question}")
            ]
        )
        
        # Chain 생성
        chain = bm25_retriever | prompt | llm | StrOutputParser() 
        
        response = chain.invoke(message) 
        print(type(response), response)
        
        # ✅ JSON 형식으로 변환
        answer_text = {"answer": response.content if hasattr(response, 'content') else str(response)}
        
        return answer_text  # 생성된 QA 체인 반환
    
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
# ✅ 데이터 시각화 기능 (그래프 생성)
@router.get("/visualize",  tags=["CHAT BOT API SERVER"])
def visualize(session_id: int, prompt : str):
    # 예제 데이터
    data = [
        {"category": "A", "value": 10},
        {"category": "B", "value": 20},
        {"category": "C", "value": 15},
        {"category": "D", "value": 25},
        {"category": "E", "value": 18},
    ]
    df = pd.DataFrame(data)

    # 바 차트 생성
    plt.figure(figsize=(6, 4))
    df.plot(kind="bar", x="category", y="value", legend=False, color="skyblue")
    plt.xlabel("Category")
    plt.ylabel("Value")
    plt.title("데이터 시각화 결과")
    plt.xticks(rotation=0)

    # 그래프를 이미지로 변환 (Base64)
    img_io = io.BytesIO()
    plt.savefig(img_io, format="png", bbox_inches="tight")
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode("utf-8")

    return {"image": f"data:image/png;base64,{img_base64}"}
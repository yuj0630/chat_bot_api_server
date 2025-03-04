from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.agents import create_react_agent, create_json_agent, AgentExecutor
from langchain.tools import create_retriever_tool # 이거 안씀
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import LLMChain # 이거 안씀
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.memory import RedisChatMessageHistory

from typing import Dict, Any, Optional, List, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field
import logging
import asyncio
from functools import lru_cache
import os
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rag_agent.log"), logging.StreamHandler()]
)
logger = logging.getLogger("rag_agent")

router = APIRouter()

# 설정 클래스
class Settings:
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1시간
    
settings = Settings()

# 열거형 정의
class AgentType(str, Enum):
    REACT = "react"
    JSON = "json"
    PLAN_AND_EXECUTE = "plan_and_execute"

class VectorStoreType(str, Enum):
    FAISS = "faiss"
    CHROMA = "chroma"

# 입력 및 출력 모델
class AgentQuery(BaseModel):
    query: str = Field(..., description="사용자의 질의 내용")
    session_id: int = Field(..., description="사용자 세션 ID")
    agent_type: AgentType = Field(default=AgentType.REACT, description="사용할 에이전트 유형")
    vector_store_type: VectorStoreType = Field(default=VectorStoreType.FAISS, description="사용할 벡터 저장소 유형")
    document_content: Optional[str] = Field(None, description="RAG 시스템에 사용될 문서 내용")
    return_source_documents: bool = Field(default=False, description="소스 문서 반환 여부")

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class AgentResponse(BaseModel):
    output: str
    intermediate_steps: Optional[List[Dict[str, Any]]] = None
    source_documents: Optional[List[SourceDocument]] = None
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

# 캐싱 도우미
@lru_cache(maxsize=100)
def get_vector_store(vector_store_id: str, vector_store_type: VectorStoreType):
    """캐싱된 벡터 스토어 인스턴스를 반환합니다."""
    logger.info(f"Vector store cache miss for {vector_store_id}. Creating new instance.")
    # 실제 구현에서는 파일 시스템이나 DB에서 저장된 벡터 스토어를 로드
    return None  # 플레이스홀더

# 문서 처리 클래스 (비동기 지원)
class DocumentProcessor:
    """문서 처리 및 임베딩 클래스"""
    
    @staticmethod
    async def load_and_split(text_content, chunk_size=settings.DEFAULT_CHUNK_SIZE, 
                    chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP):
        """텍스트를 비동기적으로 로드하고 분할"""
        def _process():
            loader = TextLoader(text_content)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            return loader.load_and_split(text_splitter)
        
        # CPU 바운드 작업을 쓰레드 풀에서 실행
        return await asyncio.to_thread(_process)
    
    @staticmethod
    async def create_vector_store(documents, vector_store_type: VectorStoreType, 
                        persist_directory: Optional[str] = None):
        """문서로부터 벡터 스토어 생성"""
        embeddings = OllamaEmbeddings()
        
        def _create_store():
            if vector_store_type == VectorStoreType.FAISS:
                return FAISS.from_documents(documents, embeddings)
            elif vector_store_type == VectorStoreType.CHROMA:
                return Chroma.from_documents(
                    documents, 
                    embeddings,
                    persist_directory=persist_directory or f"{settings.VECTOR_DB_PATH}/chroma_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
        
        return await asyncio.to_thread(_create_store)

# 향상된 RAG 도구 팩토리
class EnhancedRAGToolFactory:
    """확장 가능한 RAG 도구 팩토리 클래스"""
    
    def __init__(self, llm, callback_manager=None):
        self.llm = llm
        self.callback_manager = callback_manager or CallbackManager([])
        
    def create_tools(self, retriever, include_web_search=True, 
                include_structured_data=True):
        """확장 가능한 도구 세트 생성"""
        tools = [
            create_retriever_tool(
                retriever,
                "db_search",
                "DB에서 찾은 정보들입니다. DB 내 정보를 찾고 싶을 때 해당 검색기를 주로 이용하세요!"
            )
        ]
        
        if include_web_search:
            tools.append(create_retriever_tool(
                retriever,  # 실제 구현에서는 웹 검색 전용 retriever 구현
                "web_search",
                "웹 검색을 통해 찾은 정보들입니다. 웹사이트 내 정보를 찾고 싶을 때 해당 검색기를 주로 이용하세요!"
            ))
        
        if include_structured_data:
            tools.append(create_retriever_tool(
                retriever,  # 실제 구현에서는 RAG 전용 retriever 구현
                "RAG_search",
                "RAG 검색을 통해 찾은 정보들입니다. RAG 데이터 내 정보를 찾고 싶을 때 해당 검색기를 주로 이용하세요!"
            ))
            
        return tools
    
    def create_agent(self, tools, agent_type: AgentType):
        """지정된 유형의 에이전트 생성"""
        if agent_type == AgentType.REACT:
            return create_react_agent(self.llm, tools)
        elif agent_type == AgentType.JSON:
            return create_json_agent(self.llm, tools)
        elif agent_type == AgentType.PLAN_AND_EXECUTE:
            # 실제 구현에서는 plan and execute 에이전트 구현
            from langchain.agents.plan_and_execute.base import PlanAndExecute
            from langchain.agents.plan_and_execute.planners.base import LLMPlanner
            from langchain.agents.plan_and_execute.executors.base import LLMExecutor
            
            # 간단한 플래너 프롬프트
            planner_prompt = PromptTemplate.from_template(
                "목표: {input}\n\n이 목표를 달성하기 위한 단계별 계획을 세워주세요."
            )
            planner_chain = LLMChain(llm=self.llm, prompt=planner_prompt)
            planner = LLMPlanner(llm_chain=planner_chain)
            
            # 실행자
            executor = LLMExecutor(llm=self.llm, tools=tools)
            
            return PlanAndExecute(planner=planner, executor=executor)
        else:
            raise ValueError(f"지원되지 않는 에이전트 유형: {agent_type}")

# 세션 관리 클래스
class SessionManager:
    """세션 및 기록 관리"""
    
    @staticmethod
    def get_message_history(session_id: Union[str, int]):
        """Redis 기반 메시지 기록 반환"""
        return RedisChatMessageHistory(
            session_id=str(session_id),
            url=settings.REDIS_URL,
            ttl=settings.CACHE_TTL
        )

# 의존성 주입 도우미
async def get_llm():
    """LLM 인스턴스 제공"""
    # 실제 구현에서는 LLM 인스턴스화 로직
    from langchain.llms import Ollama
    return Ollama(model="llama3")

# 주 엔드포인트
@router.post("/response_agent", response_model=AgentResponse, tags=["CHAT BOT API SERVER"])
async def response_agent(
    query: AgentQuery,
    background_tasks: BackgroundTasks,
    llm = Depends(get_llm)
):
    """확장된 RAG 에이전트 응답 생성 엔드포인트"""
    start_time = datetime.now()
    logger.info(f"Processing query for session {query.session_id}: {query.query[:50]}...")
    
    try:
        # 문서 처리 (제공된 경우)
        if query.document_content:
            split_docs = await DocumentProcessor.load_and_split(query.document_content)
            vector_store = await DocumentProcessor.create_vector_store(
                split_docs, 
                query.vector_store_type
            )
            retriever = vector_store.as_retriever()
            
            # 백그라운드 작업으로 벡터 스토어 저장
            if query.vector_store_type == VectorStoreType.CHROMA:
                background_tasks.add_task(lambda: vector_store.persist())
        else:
            # 캐시된 벡터 스토어 사용 시도
            vector_store = get_vector_store(f"session_{query.session_id}", query.vector_store_type)
            if not vector_store:
                raise HTTPException(status_code=400, detail="문서 내용 또는 기존 벡터 스토어가 필요합니다")
            retriever = vector_store.as_retriever()
        
        # 콜백 관리자 설정
        callback_manager = CallbackManager([])  # 실제 구현에서는 적절한 콜백 추가
        
        # 도구 및 에이전트 생성
        tool_factory = EnhancedRAGToolFactory(llm, callback_manager)
        tools = tool_factory.create_tools(retriever)
        agent = tool_factory.create_agent(tools, query.agent_type)
        
        # 에이전트 실행기 설정
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        
        # 메시지 기록 관리자 가져오기
        message_history = SessionManager.get_message_history(query.session_id)
        
        # 채팅 기록이 추가된 에이전트 생성
        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: SessionManager.get_message_history(session_id),
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        # 질의에 대한 답변 생성
        response = await asyncio.to_thread(
            agent_with_chat_history.invoke,
            {"input": query.query},
            config={"configurable": {"session_id": str(query.session_id)}}
        )
        
        # 응답 구성
        source_documents = None
        if query.return_source_documents and hasattr(response, 'source_documents'):
            source_documents = [
                SourceDocument(
                    content=doc.page_content,
                    metadata=doc.metadata
                ) for doc in response.source_documents
            ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Query processed in {execution_time:.2f} seconds")
        
        return AgentResponse(
            output=response['output'],
            intermediate_steps=response.get('intermediate_steps'),
            source_documents=source_documents,
            execution_time=execution_time
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")

# 벡터 스토어 관리 엔드포인트
@router.post("/manage_vector_store", tags=["CHAT BOT API SERVER"])
async def manage_vector_store(
    operation: Literal["create", "delete", "info"],
    session_id: int,
    document_content: Optional[str] = None,
    vector_store_type: VectorStoreType = VectorStoreType.FAISS
):
    """벡터 스토어 관리 엔드포인트"""
    try:
        if operation == "create" and document_content:
            split_docs = await DocumentProcessor.load_and_split(document_content)
            vector_store = await DocumentProcessor.create_vector_store(
                split_docs, 
                vector_store_type,
                persist_directory=f"{settings.VECTOR_DB_PATH}/{session_id}"
            )
            return {"status": "success", "message": "Vector store created successfully"}
            
        elif operation == "delete":
            # 실제 구현에서는 벡터 스토어 삭제 로직
            return {"status": "success", "message": "Vector store deleted successfully"}
            
        elif operation == "info":
            # 실제 구현에서는 벡터 스토어 정보 반환 로직
            return {"status": "success", "info": {"session_id": session_id, "type": vector_store_type}}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid operation or missing document content")
            
    except Exception as e:
        logger.error(f"Vector store management error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vector store management error: {str(e)}")
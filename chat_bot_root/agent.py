from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any, Optional, Union
import re
import json
from .classified import list_all_programs, search_by_age, search_by_keyword, get_program_details

# 도구 목록 정의
tools = [
    Tool(
        name="ListAllPrograms",
        func=list_all_programs,
        description="모든 청년지원 사업 목록을 보여줍니다."
    ),
    Tool(
        name="SearchByAge",
        func=search_by_age,
        description="나이에 맞는 청년지원 사업을 찾습니다. 입력은 '25세'와 같은 형식입니다."
    ),
    Tool(
        name="SearchByKeyword",
        func=search_by_keyword,
        description="키워드로 청년지원 사업을 검색합니다. 취업, 창업, 주거 등의 키워드를 입력합니다."
    ),
    Tool(
        name="GetProgramDetails",
        func=get_program_details,
        description="특정 프로그램의 상세 정보를 제공합니다. 프로그램 번호(1, 2 등) 또는 '첫 번째' 같은 형식으로 입력합니다."
    )
]

# 프롬프트 템플릿
template = """
당신은 청년지원 사업 안내를 위한 챗봇입니다. 사용자의 질문에 친절하게 답변하고, 필요한 정보를 제공해야 합니다.

현재 대화:
{chat_history}

사용자: {input}
{agent_scratchpad}

당신은 다음 도구를 사용할 수 있습니다:

{tools}

응답 형식:
질문을 이해하고 어떤 도구를 사용할지 결정하세요.
도구를 사용해야 할 경우: Action: 도구이름, Action Input: 도구에 전달할 입력값
최종 응답: Observation: 도구 결과, Final Answer: 최종 답변

각 단계마다 사용자에게는 Final Answer만 보여집니다. Final Answer는 친절하고 대화체로 작성하세요.
"""

# 프롬프트 템플릿 클래스
class YouthSupportPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # 도구 설명 포맷팅
        tools_description = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tools"] = tools_description
        
        # 에이전트 스크래치패드 포맷팅
        if "agent_scratchpad" not in kwargs:
            kwargs["agent_scratchpad"] = ""
        
        return self.template.format(**kwargs)

# 출력 파서
class YouthSupportOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # 최종 답변 확인
        if "Final Answer:" in text:
            match = re.search(r"Final Answer: (.*)", text, re.DOTALL)
            if match:
                return AgentFinish(
                    return_values={"output": match.group(1).strip()},
                    log=text
                )
        
        # 도구 사용 확인
        match = re.search(r"Action: (.*?)[\n]*Action Input:[\s]*(.*)", text, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            return AgentAction(tool=action, tool_input=action_input, log=text)
        
        # 기본 응답
        return AgentFinish(
            return_values={"output": text.strip()},
            log=text
        )

# LangChain Agent 설정
def create_youth_support_agent(llm: BaseLLM) -> AgentExecutor:
    prompt = YouthSupportPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    
    output_parser = YouthSupportOutputParser()
    
    # LLM 체인
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # 에이전트
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )
    
    # 대화 기록을 위한 메모리
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 에이전트 실행기
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
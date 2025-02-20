# ChatBot API Server

## infomation

**ChatBot API Server**는 가시에서 제공하는 **재난관리 챗봇** 데모입니다. 
사용자가 업로드한 다양한 데이터(PDF, XLSX, CSV, HWP)를 기반으로 학습하여 **RAG 방식**으로 질문에 답합니다.

이 서버는 **FastAPI**를 기반으로 만들어졌으며, 지속적으로 작업하고 있습니다.

## 🎯  **주요 기능**

- **기본 챗봇 기능**: 사용자의 질문에 답변하고, 대화를 나눕니다.
- **RAG 방식**: 업로드한 데이터를 바탕으로 학습하고, 검색하여 더 정확한 답변을 제공합니다.
- **지원 파일 포맷**: PDF, XLSX, CSV, HWP 파일을 지원합니다.
- **빠른 응답**: FastAPI 기반으로 고성능 API 서버를 제공합니다.

## 🔧 **추후 계획**

- **Client + DB 연동**: 사용자 환경에 맞춘 데이터 저장 및 활용
- **실시간 스트리밍 응답**: 점진적으로 제공되는 챗봇 응답
- **모델 성능 향상**: 더 정교한 자연어 처리 모델 적용

## 시작하기

```bash
git clone https://github.com/yuj0630/chat_bot_api_server.git
cd chat_bot_api_server
pip install -r requirements.txt
uvicorn main:app --reload
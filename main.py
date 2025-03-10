import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

# 컨트롤 라우터 선언
from chat_bot_root import chat_bot_router
from chat_bot_root_api import api_router
from auth import auth_router

app = FastAPI(
    title="Chat Bot Server",
    version="0.1.0",
)

# cors allow path
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

# cors settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

# 기본 엔드포인트
@app.get("/")
def read_root():
    return {"message": "🚀 가시에서 제공하는 재난안전 기반 챗봇입니다. 원하는 걸 얘기해보세요!"}

# app.include_router(auth_router.router)
app.include_router(chat_bot_router.router)
app.include_router(api_router.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8892, reload=True)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# 컨트롤 라우터 선언
from chat_bot_root import chat_bot_router
from auth import auth_router

app = FastAPI(
    title="Chat Bot Server",
    version="0.0.1",
)


# app.include_router(auth_router.router)
app.include_router(chat_bot_router.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8892, reload=True)

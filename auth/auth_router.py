import os
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Security
from fastapi.security import APIKeyHeader
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Annotated
from .models import User
from .scheme import UserCreate, UserLogin, Token
from common import get_db, verify_header
from datetime import date, timedelta, datetime, timezone
import hashlib
import jwt


load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")

router = APIRouter(prefix="/api/user")


@router.get("/get_user_agent", dependencies=[verify_header()])
async def read_items(access_token: Annotated[str | None, Header()] = None):
    return {"access_token": access_token}


@router.post("/signup", tags=["Auth"])
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    already_user = db.query(User).filter(User.userid == user.userid).first()
    if already_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="User already exists"
        )

    user.password += SECRET_KEY
    user.password = hashlib.sha256(user.password.encode()).hexdigest()
    db_user = User(**user.model_dump())

    db_user.createdAt = str(date.today())
    db_user.updatedAt = str(date.today())
    # 0 = 승인안됌 | 1 = 승인 이용자 | 2 = 관리자
    db_user.authority = 0

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post("/login", tags=["Auth"])
async def login(user: UserLogin, db: Session = Depends(get_db)):
    already_user = db.query(User).filter(User.userid == user.userid).first()
    if not already_user:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="올바른 아이디를 입력해주세요"
        )

    user.password += SECRET_KEY
    user.password = hashlib.sha256(user.password.encode()).hexdigest()

    if already_user.password != user.password:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="올바른 비밀번호를 입력해주세요",
        )

    if already_user.authority == 0:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="권한이 존재하지 않습니다. 관리자 문의 바랍니다",
        )

    access_token_expires = timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
    access_token = create_access_token(
        data={"sub": already_user.userid}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

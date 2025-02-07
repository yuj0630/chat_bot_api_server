import datetime
from pydantic import BaseModel, field_validator

class Token(BaseModel):
    access_token:str
    token_type:str

class UserBase(BaseModel):
    userid: str
    username : str
    password: str

class UserLogin(BaseModel):
    userid: str
    password: str

class UserCreate(UserBase):
    pass 

class User(UserBase):
    id : int
    
    class Config:
        orm_model = True


class Token(BaseModel):
    access_token: str
    token_type: str
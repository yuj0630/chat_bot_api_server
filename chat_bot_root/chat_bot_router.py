import os
import pandas as pd
import numpy as np
import common
import json
from sqlalchemy.orm import Session
from fastapi import Depends, APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/chat_bot_root")


# @router.get("/activate_test", tags=["CHAT BOT ROOT"])
# async def get_all_culture_zones(db: Session = Depends(common.get_db)):
#     try:
#         return "hello world"
#     except Exception as e:
#         print(e)
        
@router.get("/activate_test", tags=["CHAT BOT ROOT"])
async def activate_test():
    try:
        return "hello world"
    except Exception as e:
        print(e)
        


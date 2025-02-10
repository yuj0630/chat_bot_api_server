import os
import httpx
import pandas as pd
from fastapi import HTTPException
from common.common_utils import to_dict_data, sort_value_use_key

from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# from langchain_community.vectorstores import FAISS
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser

import os
from tqdm import tqdm
from dotenv import load_dotenv

llm = ChatOllama(model="llama3.2-bllossom")

# TOY_API_SERVER = os.getenv("TOY_API_SERVER")

# async def device_count_day_util(date_from: str, date_to: str):
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.get(
#                 TOY_API_SERVER + f"/DeviceCountDay?from={date_from}&to={date_to}"
#             )
#             data = response.json()
#             df = pd.DataFrame(data)
#             return df
#     except HTTPException as e:
#         print(e + " " + "device_count_day_util")
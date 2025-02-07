import os
import httpx
import pandas as pd
from fastapi import HTTPException
from common.common_utils import to_dict_data, sort_value_use_key

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
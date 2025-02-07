import pandas as pd
from fastapi import Security
from fastapi.security import APIKeyHeader


def verify_header(access_token=Security(APIKeyHeader(name="access-token"))):
    return access_token

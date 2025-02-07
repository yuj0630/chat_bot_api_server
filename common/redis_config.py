import os
import redis
from dotenv import load_dotenv


load_dotenv()


def redis_config():
    try:
        REDIS_HOST = os.getenv("REDIS_HOST")
        REDIS_PORT = os.getenv("REDIS_PORT")
        REDIS_USERNAME = os.getenv("REDIS_USERNAME")
        REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
        REDIS_DATABASE = os.getenv("REDIS_DATABASE")
        rd = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            username=REDIS_USERNAME,
            password="admin",
        )
        return rd
    except:
        print("redis connetion failed")

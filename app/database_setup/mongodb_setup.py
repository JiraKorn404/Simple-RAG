from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI
from contextlib import asynccontextmanager
import gridfs
from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure

from app.config.config import settings

@asynccontextmanager
async def lifespan(app:FastAPI):
    global client, fs
    try:
        # test database connection
        client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print('Connected to MongoDB succesfully!')
        db = client[settings.MONGODB_NAME]
        fs = gridfs.GridFS(db)
        
        yield

        print('Closing MongoDB connection...')
        client.close()
        print('Shutdown Complete.')
    except ConnectionFailure:
        print("Failed to connect to MongoDB. Ensure MongoDB is running.")
        exit(1)

def mongodb_delete_data(
        file_id: str
):
    try:
        oid = ObjectId(file_id)
        if fs.exists(oid):
            fs.delete(oid)
            return True
        return False
    except InvalidId:
        print(f'Invalid MongoDB ID format: {file_id}')
        return False
    except Exception as e:
        print(f'Error deleting from MongoDB: {e}')
        return False
    

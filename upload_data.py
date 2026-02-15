from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# url
url="mongodb+srv://avinash:Avinash0@cluster0.zliwlkp.mongodb.net/?appName=Cluster0"

# create a new client and connect to server
client = MongoClient(url)

# Create database name and collection name
DATABASE_NAME="pwskills"
COLLECTION_NAME="waferfault"

df=pd.read_csv("D:\Projects\ML\sensor fault detection\notebooks\wafer_23012020_041211.csv")

df=df.drop("Unnamed: 0" , axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
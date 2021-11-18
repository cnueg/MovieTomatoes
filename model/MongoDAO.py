# DAO => Data Access Object
# Access => CRUD(Create, Read, Update, Delete)

from pymongo import MongoClient


# MongoDB Connection
def conn_mongo():
    client = MongoClient('localhost', 27017)  # (IP address, Port)
    db = client['local']                      # Allocating 'local' DB
    collection = db.get_collection('movie')   # Allocating 'movie' Collection
    return collection


# Create review data(데이터 등록)
def add_review(data):
    collection = conn_mongo()    # Mongo Connection
    collection.insert_one(data)  # Data save


# Select review data(데이터 조회)
def get_reviews():
    pass
from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["toxic_content_db"]
collection = db["predictions"]

def save_prediction(
    user_text: str,
    image_caption: str,
    label: str,
    score: float,
    model_name: str
):
    document = {
        "user_text": user_text,
        "image_caption": image_caption,
        "toxicity_label": label,
        "toxicity_score": score,
        "model_used": model_name,
        "created_at": datetime.utcnow()
    }
    collection.insert_one(document)

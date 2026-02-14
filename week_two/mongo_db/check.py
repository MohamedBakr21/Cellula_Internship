from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)

    # Force a connection check
    client.admin.command("ping")

    print("✅ Connected to MongoDB successfully!")

    # Optional: check database & collection
    db = client["toxic_content_db"]
    collection = db["predictions"]

    print("📂 Database:", db.name)
    print("📄 Collection:", collection.name)

except Exception as e:
    print("❌ MongoDB connection failed")
    print(e)
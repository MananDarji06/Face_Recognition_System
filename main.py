import pickle
import pandas as pd # type: ignore
from tkinter import Tk
from pymongo import MongoClient, ASCENDING # type: ignore
from bson.binary import Binary # type: ignore
from algo import FaceEngine
from app_gui import FaceRecogGUI

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "face_recog_db"
MODEL_NAME = 'facenet-pytorch'


class MongoRepo:

    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.people = self.db["people"]
        self.embeddings = self.db["embeddings"]
        self.predictions = self.db["predictions"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.people.create_index([("name", ASCENDING)], unique=True)
        self.embeddings.create_index([("person_id", ASCENDING)])
        self.predictions.create_index([("created_at", ASCENDING)])

    def insert_person(self, name: str):
        from datetime import datetime
        doc = {"name": name, "created_at": datetime.utcnow()}
        res = self.people.update_one({"name": name}, {"$setOnInsert": doc}, upsert=True)
        person = self.people.find_one({"name": name}, {"_id": 1})
        return person["_id"]

    def insert_embedding(self, person_id, embedding, source_image: str):
        from datetime import datetime
        blob = Binary(pickle.dumps(embedding))
        self.embeddings.insert_one({
            "person_id": person_id,
            "embedding": blob,
            "source_image": source_image,
            "created_at": datetime.utcnow(),
        })

    def fetch_all_embeddings(self):
        out = []
        pipeline = [
            {"$lookup": {"from": "people", "localField": "person_id", "foreignField": "_id", "as": "p"}},
            {"$unwind": "$p"},
            {"$project": {"name": "$p.name", "embedding": 1}},
        ]
        for doc in self.embeddings.aggregate(pipeline):
            name = doc["name"]
            emb = pickle.loads(bytes(doc["embedding"]))
            out.append((name, emb))
        return out

    def insert_prediction(self, image_path: str, predicted_person: str, distance: float):
        from datetime import datetime
        self.predictions.insert_one({
            "image_path": image_path,
            "predicted_person": predicted_person,
            "distance": float(distance),
            "model_name": MODEL_NAME,
            "created_at": datetime.utcnow(),
        })

    def fetch_predictions(self, limit: int = 200) -> pd.DataFrame:
        cursor = self.predictions.find({}, sort=[("created_at", -1)], limit=int(limit))
        rows = []
        for d in cursor:
            rows.append({
                "id": str(d.get("_id")),
                "image_path": d.get("image_path"),
                "predicted_person": d.get("predicted_person"),
                "distance": d.get("distance"),
                "model_name": d.get("model_name"),
                "created_at": d.get("created_at"),
            })
        return pd.DataFrame(rows)


def main():
    repo = MongoRepo()
    engine = FaceEngine()
    root = Tk(); root.geometry('900x720')
    FaceRecogGUI(root, repo=repo, engine=engine)
    root.mainloop()


if __name__ == '__main__':
    main()
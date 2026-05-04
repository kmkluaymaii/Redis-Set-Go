from messaging.broker import RedisBroker
from messaging.events import ANNOTATION_STORED, create_event
from pymongo import MongoClient

class DocumentDBService:
    def __init__(self, broker: RedisBroker, mongo_uri="mongodb://localhost:27017/", db_name="redis_set_go", collection_name="annotations"):
        self.broker = broker
        
        # Connect to MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embeddings_collection = self.db["embeddings"]

    def start(self):
        self.broker.subscribe("inference.completed", self._handle_inference_completed)
        self.broker.subscribe("embedding.created", self._handle_embedding_created)

    def _handle_inference_completed(self, event):
        """Handle an inference completion event by storing annotations."""
        image_path = event["payload"]["image_path"]
        annotations = event["payload"]["annotations"]
        
        # Store annotations in MongoDB
        document = {
            "image_path": image_path,
            "annotations": annotations
        }
        self.collection.insert_one(document)

        # Create and publish storage confirmation event
        result_event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": annotations,
            "stored_at": f"mongodb:{image_path}"
        })

        self.broker.publish(ANNOTATION_STORED, result_event)

    def get_annotations(self, image_path: str):
        """Retrieve stored annotations for an image."""
        doc = self.collection.find_one({"image_path": image_path})
        return doc["annotations"] if doc else []

    def _handle_embedding_created(self, event):
        """Handle an embedding created event by storing embeddings."""
        image_path = event["payload"]["image_path"]
        embedding = event["payload"]["embedding"]

        document = {
            "image_path": image_path,
            "embedding": embedding,
            "dimensions": len(embedding)
        }
        self.embeddings_collection.insert_one(document)

    def get_embedding(self, image_path: str):
        """Retrieve stored embedding for an image."""
        doc = self.embeddings_collection.find_one({"image_path": image_path})
        return doc["embedding"] if doc else None

    def get_all_embeddings(self):
        """Retrieve all stored embeddings."""
        embeddings = {}
        for doc in self.embeddings_collection.find():
            embeddings[doc["image_path"]] = doc["embedding"]
        return embeddings

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

    def start(self):
        self.broker.subscribe("inference.completed", self._handle_inference_completed)

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

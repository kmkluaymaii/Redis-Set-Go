from messaging.broker import RedisBroker
from messaging.events import ANNOTATION_STORED, create_event

class DocumentDBService:
    def __init__(self, broker: RedisBroker):
        self.broker = broker
        self.broker.subscribe("inference.completed", self._handle_inference_completed)
        self.storage = {}  # Mock storage: image_path -> annotations

    def _handle_inference_completed(self, event):
        """Handle an inference completion event by storing annotations."""
        image_path = event["payload"]["image_path"]
        annotations = event["payload"]["annotations"]
        
        # Store annotations
        self.storage[image_path] = annotations

        # Create and publish storage confirmation event
        result_event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": annotations,
            "stored_at": image_path
        })

        self.broker.publish(ANNOTATION_STORED, result_event)

    def get_annotations(self, image_path: str):
        """Retrieve stored annotations for an image."""
        return self.storage.get(image_path, [])

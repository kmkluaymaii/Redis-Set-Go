from messaging.broker import RedisBroker
from messaging.events import EMBEDDING_CREATED, create_event

class EmbeddingService:
    def __init__(self, broker: RedisBroker):
        self.broker = broker
        self.broker.subscribe("annotation.stored", self._handle_annotation_stored)
        self.embeddings = {}  # Mock storage: image_path -> embedding vector

    def _handle_annotation_stored(self, event):
        """Handle an annotation stored event by creating embeddings."""
        image_path = event["payload"]["image_path"]
        annotations = event["payload"]["annotations"]
        
        # Generate embedding from annotations
        embedding = self._create_embedding(image_path, annotations)
        
        # Store embedding
        self.embeddings[image_path] = embedding

        # Create and publish embedding created event
        result_event = create_event(EMBEDDING_CREATED, {
            "image_path": image_path,
            "embedding": embedding,
            "dimensions": len(embedding)
        })

        self.broker.publish(EMBEDDING_CREATED, result_event)

    def _create_embedding(self, image_path: str, annotations: list) -> list:
        """Create a mock embedding vector from annotations."""
        # Simple mock: create a vector based on annotation labels
        base_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        if annotations:
            # Modify vector based on first annotation
            label_hash = hash(annotations[0].get("label", "unknown")) % 100
            base_vector[0] += label_hash * 0.001
        return base_vector

    def get_embedding(self, image_path: str):
        """Retrieve stored embedding for an image."""
        return self.embeddings.get(image_path)

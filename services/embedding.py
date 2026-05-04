from messaging.broker import RedisBroker
from messaging.events import EMBEDDING_CREATED, create_event
from PIL import Image
import hashlib

class EmbeddingService:
    def __init__(self, broker: RedisBroker):
        self.broker = broker
        self.embeddings = {}  # Mock storage: image_path -> embedding vector

    def start(self):
        self.broker.subscribe("annotation.stored", self._handle_annotation_stored)

    def _handle_annotation_stored(self, event):
        """Handle an annotation stored event by creating embeddings."""
        image_path = event["payload"]["image_path"]
        annotations = event["payload"]["annotations"]
        
        # Generate embedding from image and annotations
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
        """Create a simulated embedding vector from image and annotations."""
        try:
            # Load image and compute a simple hash-based embedding
            with Image.open(image_path) as img:
                # Get image data
                img_data = img.tobytes()
                # Create hash
                img_hash = hashlib.md5(img_data).hexdigest()
                
                # Convert hash to numerical vector (128 dimensions)
                embedding = []
                for i in range(0, 32, 2):  # 16 values from 32 chars
                    hex_pair = img_hash[i:i+2]
                    val = int(hex_pair, 16) / 255.0  # Normalize to 0-1
                    embedding.append(val)
                
                # Extend to 256 dimensions by repeating and modifying
                while len(embedding) < 256:
                    embedding.extend([x * 0.9 for x in embedding[:len(embedding)]])
                embedding = embedding[:256]
                
                # Modify based on annotations
                if annotations:
                    for i, ann in enumerate(annotations[:5]):  # Use first 5 annotations
                        label_hash = hash(ann.get("label", "unknown")) % 1000
                        confidence = ann.get("confidence", 0.5)
                        # Modify embedding based on label and confidence
                        for j in range(min(10, len(embedding))):
                            embedding[j] += (label_hash * 0.0001 * confidence)
                            embedding[j] = min(1.0, max(0.0, embedding[j]))  # Clamp
                
                return embedding
                
        except Exception as e:
            print(f"Embedding error: {e}")
            # Fallback to simple vector
            return [0.1 + i * 0.01 for i in range(256)]

    def get_embedding(self, image_path: str):
        """Retrieve stored embedding for an image."""
        return self.embeddings.get(image_path)

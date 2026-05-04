from messaging.broker import RedisBroker
from messaging.events import EMBEDDING_CREATED, create_event
from PIL import Image
import hashlib
import faiss
import numpy as np

class EmbeddingService:
    def __init__(self, broker: RedisBroker, embedding_dim=256):
        self.broker = broker
        self.embedding_dim = embedding_dim

        # Initialize FAISS index for L2 distance (cosine similarity)
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Keep track of image paths corresponding to embeddings in the index
        self.image_paths = []

        # For backward compatibility, maintain a dictionary mapping
        self.embeddings = {}

    def start(self):
        self.broker.subscribe("annotation.stored", self._handle_annotation_stored)
        # Load existing embeddings from database
        self._load_existing_embeddings()

    def _load_existing_embeddings(self):
        """Load existing embeddings from MongoDB into FAISS index."""
        try:
            from services.document_db import DocumentDBService

            temp_db = DocumentDBService(self.broker)
            existing_embeddings = temp_db.get_all_embeddings()

            for image_path, embedding in existing_embeddings.items():
                self.embeddings[image_path] = embedding

                embedding_np = np.array([embedding], dtype=np.float32)
                self.index.add(embedding_np)
                self.image_paths.append(image_path)

            print(f"Loaded {len(existing_embeddings)} existing embeddings into FAISS index")

        except Exception as e:
            print(f"Warning: Could not load existing embeddings: {e}")
    
    def _handle_annotation_stored(self, event):
        """Handle an annotation stored event by creating embeddings."""
        image_path = event["payload"]["image_path"]
        annotations = event["payload"]["annotations"]

        # Generate embedding from image and annotations
        embedding = self._create_embedding(image_path, annotations)

        # Store embedding in both dictionary (for backward compatibility) and FAISS index
        self.embeddings[image_path] = embedding

        # Add to FAISS index
        embedding_np = np.array([embedding], dtype=np.float32)
        self.index.add(embedding_np)
        self.image_paths.append(image_path)

        # Create and publish embedding created event
        result_event = create_event("embedding.created", {
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

    def search_similar(self, query_embedding: list, k: int = 5) -> list:
        """Search for k most similar embeddings using FAISS."""
        if self.index.ntotal == 0:
            return []

        # Convert query to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)

        # Search for k nearest neighbors
        distances, indices = self.index.search(query_np, min(k, self.index.ntotal))

        # Return list of (image_path, distance) tuples
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(distances[0][i])))

        return results

    def get_total_embeddings(self) -> int:
        """Get total number of embeddings in the FAISS index."""
        return self.index.ntotal

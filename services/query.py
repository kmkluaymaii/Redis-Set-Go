from messaging.events import QUERY_SUBMITTED, QUERY_COMPLETED, create_event
import hashlib
import numpy as np

class QueryService:
    def __init__(self, broker, embedding_service=None):
        self.broker = broker
        self.embedding_service = embedding_service

    def start(self):
        self.broker.subscribe(QUERY_SUBMITTED, self._handle_query_submitted)

    def _handle_query_submitted(self, event):
        query_text = event["payload"]["query"]
        results = self._search_embeddings(query_text)

        result_event = create_event(QUERY_COMPLETED, {
            "query": query_text,
            "results": results
        })
        self.broker.publish(QUERY_COMPLETED, result_event)

    def _create_query_embedding(self, query_text: str) -> list:
        """Create an embedding vector from query text using the same method as images."""
        # Create hash from query text
        query_hash = hashlib.md5(query_text.encode()).hexdigest()

        # Convert hash to numerical vector (same as image embeddings)
        embedding = []
        for i in range(0, 32, 2):  # 16 values from 32 chars
            hex_pair = query_hash[i:i+2]
            val = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(val)

        # Extend to 256 dimensions by repeating and modifying
        while len(embedding) < 256:
            embedding.extend([x * 0.9 for x in embedding[:len(embedding)]])
        embedding = embedding[:256]

        return embedding

    def _search_embeddings(self, query_text: str):
        if not self.embedding_service:
            return []

        # Create query embedding
        query_embedding = self._create_query_embedding(query_text)

        # Search for similar embeddings using FAISS
        similar_results = self.embedding_service.search_similar(query_embedding, k=10)

        # Format results
        results = []
        for image_path, distance in similar_results:
            # Convert L2 distance to similarity score (higher is better)
            similarity_score = max(0, 1.0 - distance / 10.0)  # Normalize distance

            results.append({
                "image_path": image_path,
                "similarity_score": similarity_score,
                "distance": distance
            })

        return results

    def get_results_for_query(self, query_text: str):
        return self._search_embeddings(query_text)

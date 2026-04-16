from messaging.events import QUERY_SUBMITTED, QUERY_COMPLETED, create_event

class QueryService:
    def __init__(self, broker, embedding_store=None):
        self.broker = broker
        self.embedding_store = embedding_store if embedding_store is not None else {}
        self.broker.subscribe(QUERY_SUBMITTED, self._handle_query_submitted)

    def _handle_query_submitted(self, event):
        query_text = event["payload"]["query"]
        results = self._search_embeddings(query_text)

        result_event = create_event(QUERY_COMPLETED, {
            "query": query_text,
            "results": results
        })
        self.broker.publish(QUERY_COMPLETED, result_event)

    def _search_embeddings(self, query_text: str):
        if not self.embedding_store:
            return []

        # Return all stored embeddings with a simple match score placeholder
        return [
            {
                "image_path": image_path,
                "embedding_length": len(embedding),
                "score": 1.0
            }
            for image_path, embedding in self.embedding_store.items()
        ]

    def get_results_for_query(self, query_text: str):
        return self._search_embeddings(query_text)

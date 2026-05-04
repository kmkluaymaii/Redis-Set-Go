import unittest
from unittest.mock import MagicMock
from services.query import QueryService
from messaging.events import QUERY_SUBMITTED, QUERY_COMPLETED, create_event


class TestQueryService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        # Real QueryService accepts `embedding_service=`, not `embedding_store=`.
        # subscribe() is called in start(), not __init__.
        self.mock_embedding_service = MagicMock()
        self.mock_embedding_service.search_similar.return_value = [
            ("/image/1.jpg", 0.05),
            ("/image/2.jpg", 0.20),
        ]
        self.service = QueryService(
            self.mock_broker,
            embedding_service=self.mock_embedding_service,
        )

    def test_constructor_subscribes_to_topic(self):
        """start() must subscribe to QUERY_SUBMITTED."""
        self.service.start()
        self.mock_broker.subscribe.assert_called_once_with(
            QUERY_SUBMITTED, self.service._handle_query_submitted
        )

    def test_handle_query_submitted_publishes_results(self):
        """_handle_query_submitted must publish a QUERY_COMPLETED event."""
        query_text = "Find cat images"
        event = create_event(QUERY_SUBMITTED, {"query": query_text})

        self.service._handle_query_submitted(event)

        self.mock_broker.publish.assert_called_once()
        args = self.mock_broker.publish.call_args
        self.assertEqual(args[0][0], QUERY_COMPLETED)

        published_event = args[0][1]
        self.assertEqual(published_event["topic"], QUERY_COMPLETED)
        self.assertEqual(published_event["payload"]["query"], query_text)
        # Two results are returned by the mock embedding service
        self.assertEqual(len(published_event["payload"]["results"]), 2)

    def test_get_results_for_query_returns_embeddings(self):
        """get_results_for_query must return one dict per similar image."""
        results = self.service.get_results_for_query("search")

        self.assertEqual(len(results), 2)
        # Results are ordered by distance ascending (closest first)
        image_paths = [r["image_path"] for r in results]
        self.assertIn("/image/1.jpg", image_paths)
        self.assertIn("/image/2.jpg", image_paths)
        for r in results:
            self.assertIn("similarity_score", r)
            self.assertIn("distance", r)

    def test_search_embeddings_empty_store(self):
        """When embedding_service returns no results, get_results_for_query must return []."""
        self.mock_embedding_service.search_similar.return_value = []
        self.assertEqual(self.service.get_results_for_query("search"), [])

    def test_search_embeddings_no_embedding_service(self):
        """When no embedding_service is provided, get_results_for_query must return []."""
        service = QueryService(self.mock_broker)
        self.assertEqual(service.get_results_for_query("search"), [])

    def test_handle_query_submitted_calls_search_similar(self):
        """_handle_query_submitted must delegate to embedding_service.search_similar."""
        event = create_event(QUERY_SUBMITTED, {"query": "dog"})
        self.service._handle_query_submitted(event)
        self.mock_embedding_service.search_similar.assert_called_once()


if __name__ == "__main__":
    unittest.main()
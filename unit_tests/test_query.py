import unittest
from unittest.mock import MagicMock
from services.query import QueryService
from messaging.events import QUERY_SUBMITTED, QUERY_COMPLETED, create_event

class TestQueryService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        self.embedding_store = {
            "/image/1.jpg": [0.1, 0.2, 0.3],
            "/image/2.jpg": [0.2, 0.3, 0.4]
        }
        self.service = QueryService(self.mock_broker, embedding_store=self.embedding_store)

    def test_constructor_subscribes_to_topic(self):
        self.mock_broker.subscribe.assert_called_once_with(QUERY_SUBMITTED, self.service._handle_query_submitted)

    def test_handle_query_submitted_publishes_results(self):
        query_text = "Find cat images"
        event = create_event(QUERY_SUBMITTED, {"query": query_text})

        self.service._handle_query_submitted(event)

        self.mock_broker.publish.assert_called_once()
        args = self.mock_broker.publish.call_args
        self.assertEqual(args[0][0], QUERY_COMPLETED)

        published_event = args[0][1]
        self.assertEqual(published_event["topic"], QUERY_COMPLETED)
        self.assertEqual(published_event["payload"]["query"], query_text)
        self.assertEqual(len(published_event["payload"]["results"]), 2)

    def test_get_results_for_query_returns_embeddings(self):
        results = self.service.get_results_for_query("search")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["image_path"], "/image/1.jpg")
        self.assertEqual(results[1]["image_path"], "/image/2.jpg")

    def test_search_embeddings_empty_store(self):
        service = QueryService(self.mock_broker, embedding_store={})
        self.mock_broker.subscribe.assert_called_with(QUERY_SUBMITTED, service._handle_query_submitted)
        self.assertEqual(service.get_results_for_query("search"), [])

if __name__ == '__main__':
    unittest.main()

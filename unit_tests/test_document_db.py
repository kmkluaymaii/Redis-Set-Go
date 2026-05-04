import unittest
from unittest.mock import MagicMock, patch
from messaging.events import INFERENCE_COMPLETED, ANNOTATION_STORED, create_event


# Patch MongoClient at the module level so DocumentDBService.__init__ never
# opens a real TCP connection to localhost:27017.
@patch("services.document_db.MongoClient", autospec=True)
class TestDocumentDBService(unittest.TestCase):

    def _make_service(self, mock_mongo_cls):
        """Build a DocumentDBService with a fully mocked Mongo stack."""
        from services.document_db import DocumentDBService

        self.mock_collection = MagicMock()
        self.mock_embeddings_collection = MagicMock()

        mock_client = mock_mongo_cls.return_value
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.side_effect = lambda name: (
            self.mock_collection if name == "annotations" else self.mock_embeddings_collection
        )

        service = DocumentDBService(MagicMock())
        # Inject mocks directly so every method call goes to our mocks.
        service.collection = self.mock_collection
        service.embeddings_collection = self.mock_embeddings_collection
        return service

    # ------------------------------------------------------------------
    # Each test receives mock_mongo_cls as its first extra argument
    # because of the class-level @patch decorator.
    # ------------------------------------------------------------------

    def test_start_subscribes_to_topic(self, mock_mongo_cls):
        """start() must subscribe to both inference.completed and embedding.created."""
        service = self._make_service(mock_mongo_cls)
        mock_broker = MagicMock()
        service.broker = mock_broker

        service.start()

        # start() registers two subscriptions; verify the critical one is present
        mock_broker.subscribe.assert_any_call(
            "inference.completed", service._handle_inference_completed
        )
        mock_broker.subscribe.assert_any_call(
            "embedding.created", service._handle_embedding_created
        )
        self.assertEqual(mock_broker.subscribe.call_count, 2)

    def test_handle_inference_completed(self, mock_mongo_cls):
        """inference.completed events must be stored and annotation.stored published."""
        service = self._make_service(mock_mongo_cls)
        mock_broker = MagicMock()
        service.broker = mock_broker

        image_path = "/path/to/test/image.jpg"
        annotations = [{"label": "object", "confidence": 0.9}]
        event = create_event(INFERENCE_COMPLETED, {
            "image_path": image_path,
            "annotations": annotations,
        })

        service._handle_inference_completed(event)

        # MongoDB insert must have been called once with the right document
        self.mock_collection.insert_one.assert_called_once()
        inserted_doc = self.mock_collection.insert_one.call_args[0][0]
        self.assertEqual(inserted_doc["image_path"], image_path)
        self.assertEqual(inserted_doc["annotations"], annotations)

        # Must publish annotation.stored
        mock_broker.publish.assert_called_once()
        args = mock_broker.publish.call_args
        self.assertEqual(args[0][0], ANNOTATION_STORED)

        published_event = args[0][1]
        self.assertEqual(published_event["topic"], ANNOTATION_STORED)
        self.assertEqual(published_event["payload"]["image_path"], image_path)
        self.assertEqual(published_event["payload"]["annotations"], annotations)
        self.assertEqual(published_event["payload"]["stored_at"], f"mongodb:{image_path}")

    def test_storage_and_retrieval(self, mock_mongo_cls):
        """Annotations stored via an event must be retrievable via get_annotations."""
        service = self._make_service(mock_mongo_cls)
        service.broker = MagicMock()

        image_path = "/test/image.jpg"
        annotations = [{"label": "cat", "confidence": 0.8}]
        event = create_event(INFERENCE_COMPLETED, {
            "image_path": image_path,
            "annotations": annotations,
        })

        # Teach find_one to return the stored document
        self.mock_collection.find_one.return_value = {
            "image_path": image_path,
            "annotations": annotations,
        }

        service._handle_inference_completed(event)
        retrieved = service.get_annotations(image_path)

        self.mock_collection.find_one.assert_called_once_with({"image_path": image_path})
        self.assertEqual(retrieved, annotations)

    def test_get_annotations_nonexistent(self, mock_mongo_cls):
        """get_annotations for an unknown path must return an empty list."""
        service = self._make_service(mock_mongo_cls)
        self.mock_collection.find_one.return_value = None

        retrieved = service.get_annotations("/nonexistent.jpg")
        self.assertEqual(retrieved, [])


if __name__ == "__main__":
    unittest.main()
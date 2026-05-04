import unittest
from unittest.mock import MagicMock, patch

from messaging.events import (
    INFERENCE_COMPLETED,
    ANNOTATION_STORED,
    create_event,
)

from services.document_db import DocumentDBService


class TestDocumentDBService(unittest.TestCase):

    def setUp(self):
        # Patch MongoClient safely per-test (avoids autospec collisions)
        self.mongo_patcher = patch("services.document_db.MongoClient")
        self.mock_mongo_cls = self.mongo_patcher.start()
        self.addCleanup(self.mongo_patcher.stop)

        # Create mocked DB collections
        self.mock_collection = MagicMock()
        self.mock_embeddings_collection = MagicMock()

        # Configure mock Mongo client → db → collections
        mock_client = self.mock_mongo_cls.return_value
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        def get_collection(name):
            if name == "annotations":
                return self.mock_collection
            return self.mock_embeddings_collection

        mock_db.__getitem__.side_effect = get_collection

        # Create service
        self.service = DocumentDBService(MagicMock())
        self.service.collection = self.mock_collection
        self.service.embeddings_collection = self.mock_embeddings_collection

        # Mock broker
        self.service.broker = MagicMock()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_start_subscribes_to_topic(self):
        """start() must subscribe to inference.completed and embedding.created."""
        self.service.start()

        self.service.broker.subscribe.assert_any_call(
            "inference.completed",
            self.service._handle_inference_completed,
        )
        self.service.broker.subscribe.assert_any_call(
            "embedding.created",
            self.service._handle_embedding_created,
        )
        self.assertEqual(self.service.broker.subscribe.call_count, 2)

    def test_handle_inference_completed(self):
        """Event must be stored and annotation.stored published."""
        image_path = "/path/to/test/image.jpg"
        annotations = [{"label": "object", "confidence": 0.9}]

        event = create_event(INFERENCE_COMPLETED, {
            "image_path": image_path,
            "annotations": annotations,
        })

        self.service._handle_inference_completed(event)

        # Verify DB insert
        self.mock_collection.insert_one.assert_called_once()
        inserted_doc = self.mock_collection.insert_one.call_args[0][0]

        self.assertEqual(inserted_doc["image_path"], image_path)
        self.assertEqual(inserted_doc["annotations"], annotations)

        # Verify event publish
        self.service.broker.publish.assert_called_once()

        topic, published_event = self.service.broker.publish.call_args[0]
        self.assertEqual(topic, ANNOTATION_STORED)

        self.assertEqual(published_event["topic"], ANNOTATION_STORED)
        self.assertEqual(published_event["payload"]["image_path"], image_path)
        self.assertEqual(published_event["payload"]["annotations"], annotations)
        self.assertEqual(
            published_event["payload"]["stored_at"],
            f"mongodb:{image_path}",
        )

    def test_storage_and_retrieval(self):
        """Stored annotations must be retrievable."""
        image_path = "/test/image.jpg"
        annotations = [{"label": "cat", "confidence": 0.8}]

        event = create_event(INFERENCE_COMPLETED, {
            "image_path": image_path,
            "annotations": annotations,
        })

        self.mock_collection.find_one.return_value = {
            "image_path": image_path,
            "annotations": annotations,
        }

        self.service._handle_inference_completed(event)
        retrieved = self.service.get_annotations(image_path)

        self.mock_collection.find_one.assert_called_once_with(
            {"image_path": image_path}
        )
        self.assertEqual(retrieved, annotations)

    def test_get_annotations_nonexistent(self):
        """Unknown image path must return empty list."""
        self.mock_collection.find_one.return_value = None

        retrieved = self.service.get_annotations("/nonexistent.jpg")
        self.assertEqual(retrieved, [])


if __name__ == "__main__":
    unittest.main()
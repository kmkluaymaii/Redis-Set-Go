import unittest
from unittest.mock import MagicMock
from services.embedding import EmbeddingService
from messaging.events import ANNOTATION_STORED, EMBEDDING_CREATED, create_event

class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        self.service = EmbeddingService(self.mock_broker)

    def test_constructor_subscribes_to_topic(self):
        """Test that the service subscribes to annotation.stored on initialization."""
        self.mock_broker.subscribe.assert_called_once_with("annotation.stored", self.service._handle_annotation_stored)

    def test_handle_annotation_stored(self):
        """Test that annotation stored events are processed and embedding created events are published."""
        # Create a mock annotation stored event
        image_path = "/path/to/test/image.jpg"
        annotations = [{"label": "object", "confidence": 0.9}]
        event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": annotations,
            "stored_at": image_path
        })

        # Handle the event
        self.service._handle_annotation_stored(event)

        # Verify that publish was called
        self.mock_broker.publish.assert_called_once()
        args = self.mock_broker.publish.call_args
        self.assertEqual(args[0][0], EMBEDDING_CREATED)

        # Check the published event structure
        published_event = args[0][1]
        self.assertEqual(published_event["topic"], EMBEDDING_CREATED)
        self.assertEqual(published_event["payload"]["image_path"], image_path)
        self.assertIn("embedding", published_event["payload"])
        self.assertIn("dimensions", published_event["payload"])
        self.assertEqual(published_event["payload"]["dimensions"], 5)  # Our mock vector length

    def test_create_embedding_with_annotations(self):
        """Test embedding creation with annotations."""
        annotations = [{"label": "cat", "confidence": 0.9}]
        embedding = self.service._create_embedding("/test.jpg", annotations)
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 5)
        self.assertTrue(all(isinstance(x, float) for x in embedding))

    def test_create_embedding_empty_annotations(self):
        """Test embedding creation with empty annotations."""
        embedding = self.service._create_embedding("/test.jpg", [])
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 5)
        # Should return base vector
        self.assertEqual(embedding, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_storage_and_retrieval(self):
        """Test that embeddings are stored and can be retrieved."""
        image_path = "/test/image.jpg"
        annotations = [{"label": "dog", "confidence": 0.8}]

        # Simulate storing via event handling
        event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": annotations,
            "stored_at": image_path
        })
        self.service._handle_annotation_stored(event)

        # Test retrieval
        retrieved = self.service.get_embedding(image_path)
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved, list)
        self.assertEqual(len(retrieved), 5)

    def test_get_embedding_nonexistent(self):
        """Test retrieving embedding for non-existent image returns None."""
        retrieved = self.service.get_embedding("/nonexistent.jpg")
        self.assertIsNone(retrieved)

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import MagicMock
from services.document_db import DocumentDBService
from messaging.events import INFERENCE_COMPLETED, ANNOTATION_STORED, create_event

class TestDocumentDBService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        self.service = DocumentDBService(self.mock_broker)

    def test_constructor_subscribes_to_topic(self):
        """Test that the service subscribes to inference.completed on initialization."""
        self.mock_broker.subscribe.assert_called_once_with("inference.completed", self.service._handle_inference_completed)

    def test_handle_inference_completed(self):
        """Test that inference completion events are processed and annotation stored events are published."""
        # Create a mock inference completed event
        image_path = "/path/to/test/image.jpg"
        annotations = [{"label": "object", "confidence": 0.9}]
        event = create_event(INFERENCE_COMPLETED, {
            "image_path": image_path,
            "annotations": annotations
        })

        # Handle the event
        self.service._handle_inference_completed(event)

        # Verify that publish was called
        self.mock_broker.publish.assert_called_once()
        args = self.mock_broker.publish.call_args
        self.assertEqual(args[0][0], ANNOTATION_STORED)

        # Check the published event structure
        published_event = args[0][1]
        self.assertEqual(published_event["topic"], ANNOTATION_STORED)
        self.assertEqual(published_event["payload"]["image_path"], image_path)
        self.assertEqual(published_event["payload"]["annotations"], annotations)
        self.assertEqual(published_event["payload"]["stored_at"], image_path)

    def test_storage_and_retrieval(self):
        """Test that annotations are stored and can be retrieved."""
        image_path = "/test/image.jpg"
        annotations = [{"label": "cat", "confidence": 0.8}]

        # Simulate storing via event handling
        event = create_event(INFERENCE_COMPLETED, {
            "image_path": image_path,
            "annotations": annotations
        })
        self.service._handle_inference_completed(event)

        # Test retrieval
        retrieved = self.service.get_annotations(image_path)
        self.assertEqual(retrieved, annotations)

    def test_get_annotations_nonexistent(self):
        """Test retrieving annotations for non-existent image returns empty list."""
        retrieved = self.service.get_annotations("/nonexistent.jpg")
        self.assertEqual(retrieved, [])

if __name__ == '__main__':
    unittest.main()

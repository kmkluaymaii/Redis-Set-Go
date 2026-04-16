import unittest
from unittest.mock import MagicMock
from services.inference import InferenceService
from messaging.events import IMAGE_SUBMITTED, INFERENCE_COMPLETED, create_event

class TestInferenceService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        self.service = InferenceService(self.mock_broker)

    def test_constructor_subscribes_to_topic(self):
        """Test that the service subscribes to image.submitted on initialization."""
        self.mock_broker.subscribe.assert_called_once_with("image.submitted", self.service._handle_image_submitted)

    def test_handle_image_submitted(self):
        """Test that image submission events are processed and inference completed events are published."""
        # Create a mock image submitted event
        image_path = "/path/to/test/image.jpg"
        event = create_event(IMAGE_SUBMITTED, {
            "stored_path": image_path,
            "filename": "test.jpg"
        })

        # Handle the event
        self.service._handle_image_submitted(event)

        # Verify that publish was called
        self.mock_broker.publish.assert_called_once()
        args = self.mock_broker.publish.call_args
        self.assertEqual(args[0][0], INFERENCE_COMPLETED)

        # Check the published event structure
        published_event = args[0][1]
        self.assertEqual(published_event["topic"], INFERENCE_COMPLETED)
        self.assertEqual(published_event["payload"]["image_path"], image_path)
        self.assertIn("annotations", published_event["payload"])

    def test_run_inference_basic_results(self):
        """Test that _run_inference returns basic annotation data."""
        results = self.service._run_inference("/fake/path.jpg")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "object")
        self.assertEqual(results[0]["confidence"], 0.9)

if __name__ == '__main__':
    unittest.main()

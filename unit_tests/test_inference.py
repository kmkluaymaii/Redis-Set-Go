import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call
from services.inference import InferenceService
from messaging.events import IMAGE_SUBMITTED, INFERENCE_COMPLETED, create_event

class TestInferenceService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        self.service = InferenceService(self.mock_broker)

    def test_start_service(self):
        """Test that starting the service subscribes to the correct topic."""
        self.service.start()
        self.assertTrue(self.service.running)
        self.mock_broker.subscribe.assert_called_once_with("image.submitted", self.service._handle_image_submitted)

    def test_stop_service(self):
        """Test that stopping the service sets running to False."""
        self.service.start()
        self.service.stop()
        self.assertFalse(self.service.running)

    def test_handle_image_submitted_success(self):
        """Test successful processing of an image submission event."""
        # Create a mock image submitted event
        image_path = "/path/to/test/image.jpg"
        event = create_event(IMAGE_SUBMITTED, {
            "stored_path": image_path,
            "filename": "test.jpg"
        })

        # Start the service and handle the event
        self.service.start()
        self.service._handle_image_submitted(event)

        # Verify that publish was called
        self.assertEqual(self.mock_broker.publish.call_count, 1)
        args = self.mock_broker.publish.call_args
        self.assertEqual(args[0][0], INFERENCE_COMPLETED)

        # Check the published event structure
        published_event = args[0][1]
        self.assertEqual(published_event["topic"], INFERENCE_COMPLETED)
        self.assertEqual(published_event["payload"]["image_path"], image_path)
        self.assertIn("annotations", published_event["payload"])
        self.assertIn("confidence", published_event["payload"])
        self.assertEqual(published_event["payload"]["model_version"], "v1.0")

        # Check that annotations contain expected mock data
        annotations = published_event["payload"]["annotations"]
        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[0]["label"], "cat")
        self.assertEqual(annotations[1]["label"], "dog")

    def test_handle_image_submitted_when_stopped(self):
        """Test that events are ignored when service is stopped."""
        event = create_event(IMAGE_SUBMITTED, {"stored_path": "/test.jpg"})

        # Don't start the service
        self.service._handle_image_submitted(event)

        # Should not publish anything
        self.mock_broker.publish.assert_not_called()

    def test_handle_image_submitted_with_error(self):
        """Test error handling in image processing."""
        # Create an event with invalid payload (missing stored_path)
        event = create_event(IMAGE_SUBMITTED, {"filename": "test.jpg"})

        self.service.start()

        # This should not crash, but handle the error gracefully
        with patch('builtins.print') as mock_print:
            self.service._handle_image_submitted(event)
            # Should print an error message
            mock_print.assert_called()

    def test_run_inference_mock_results(self):
        """Test that _run_inference returns expected mock data."""
        results = self.service._run_inference("/fake/path.jpg")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

        # Check first annotation
        cat_annotation = results[0]
        self.assertEqual(cat_annotation["label"], "cat")
        self.assertEqual(cat_annotation["confidence"], 0.92)
        self.assertEqual(cat_annotation["bbox"], [0.1, 0.2, 0.8, 0.9])

        # Check second annotation
        dog_annotation = results[1]
        self.assertEqual(dog_annotation["label"], "dog")
        self.assertEqual(dog_annotation["confidence"], 0.87)
        self.assertEqual(dog_annotation["bbox"], [0.3, 0.1, 0.7, 0.8])

if __name__ == '__main__':
    unittest.main()

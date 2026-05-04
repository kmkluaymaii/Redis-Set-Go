import os
import unittest
from unittest.mock import MagicMock, patch
from services.inference import InferenceService
from messaging.events import IMAGE_SUBMITTED, INFERENCE_COMPLETED, create_event


class TestInferenceService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        self.service = InferenceService(self.mock_broker)

    def test_constructor_subscribes_to_topic(self):
        """Test that start() subscribes to image.submitted."""
        self.service.start()
        self.mock_broker.subscribe.assert_called_once_with(
            "image.submitted", self.service._handle_image_submitted
        )

    def test_handle_image_submitted(self):
        """Test that image submission events are processed and inference.completed is published."""
        image_path = os.path.join("images", "dog.jpg")
        event = create_event(IMAGE_SUBMITTED, {
            "stored_path": image_path,
            "filename": "dog.jpg",
        })

        # Patch _run_inference so no real file I/O is needed
        fake_annotations = [
            {"label": "Dog", "confidence": 0.95, "bbox": [10, 10, 200, 200], "animal": "Dog"}
        ]
        with patch.object(self.service, "_run_inference", return_value=fake_annotations):
            self.service._handle_image_submitted(event)

        self.mock_broker.publish.assert_called_once()
        args = self.mock_broker.publish.call_args
        self.assertEqual(args[0][0], INFERENCE_COMPLETED)

        published_event = args[0][1]
        self.assertEqual(published_event["topic"], INFERENCE_COMPLETED)
        self.assertEqual(published_event["payload"]["image_path"], image_path)
        self.assertIn("annotations", published_event["payload"])
        self.assertEqual(published_event["payload"]["annotations"], fake_annotations)

    def test_run_inference_fallback_on_bad_path(self):
        """Test that _run_inference returns a single fallback annotation when the file doesn't exist."""
        results = self.service._run_inference("/nonexistent/path/image.jpg")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "landmark")
        self.assertEqual(results[0]["confidence"], 0.9)
        self.assertEqual(results[0]["bbox"], [0, 0, 100, 100])

    def test_run_inference_returns_annotations_for_known_animal(self):
        """Test that _run_inference returns properly structured annotations for a known animal image."""
        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.size = (640, 480)

        with patch("PIL.Image.open", return_value=mock_img):
            results = self.service._run_inference("/fake/images/dog.jpg")

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for annotation in results:
            self.assertIn("label", annotation)
            self.assertIn("confidence", annotation)
            self.assertIn("bbox", annotation)
            self.assertIsInstance(annotation["bbox"], list)
            self.assertEqual(len(annotation["bbox"]), 4)
            self.assertGreaterEqual(annotation["confidence"], 0.0)
            self.assertLessEqual(annotation["confidence"], 1.0)

    def test_run_inference_returns_annotations_for_unknown_animal(self):
        """Test that _run_inference falls back gracefully for an unrecognised filename."""
        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.size = (800, 600)

        with patch("PIL.Image.open", return_value=mock_img):
            results = self.service._run_inference("/fake/images/unicorn.jpg")

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for annotation in results:
            self.assertIn("label", annotation)
            self.assertIn("confidence", annotation)
            self.assertIn("bbox", annotation)

    def test_run_inference_annotation_bbox_within_image_bounds(self):
        """Test that all bounding boxes are within the image dimensions."""
        width, height = 640, 480
        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.size = (width, height)

        with patch("PIL.Image.open", return_value=mock_img):
            results = self.service._run_inference("/fake/images/cat.jpg")

        for annotation in results:
            x1, y1, x2, y2 = annotation["bbox"]
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, width)
            self.assertLessEqual(y2, height)


if __name__ == "__main__":
    unittest.main()
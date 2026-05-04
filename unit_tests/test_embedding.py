import unittest
from unittest.mock import MagicMock, patch
from services.embedding import EmbeddingService
from messaging.events import ANNOTATION_STORED, EMBEDDING_CREATED, create_event
import os


class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        # Patch _load_existing_embeddings so start() doesn't hit MongoDB
        with patch.object(EmbeddingService, "_load_existing_embeddings", return_value=None):
            self.service = EmbeddingService(self.mock_broker)

    def test_constructor_subscribes_to_topic(self):
        """Test that start() subscribes to annotation.stored."""
        with patch.object(self.service, "_load_existing_embeddings", return_value=None):
            self.service.start()
        self.mock_broker.subscribe.assert_called_once_with(
            "annotation.stored", self.service._handle_annotation_stored
        )

    def test_handle_annotation_stored(self):
        """Test that annotation stored events are processed and embedding.created is published."""
        image_path = "/path/to/test/image.jpg"
        annotations = [{"label": "object", "confidence": 0.9}]
        event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": annotations,
            "stored_at": image_path,
        })

        # Patch _create_embedding to avoid needing a real image file
        fake_embedding = [0.1] * 256
        with patch.object(self.service, "_create_embedding", return_value=fake_embedding):
            self.service._handle_annotation_stored(event)

        self.mock_broker.publish.assert_called_once()
        args = self.mock_broker.publish.call_args
        self.assertEqual(args[0][0], EMBEDDING_CREATED)

        published_event = args[0][1]
        self.assertEqual(published_event["topic"], EMBEDDING_CREATED)
        self.assertEqual(published_event["payload"]["image_path"], image_path)
        self.assertIn("embedding", published_event["payload"])
        self.assertIn("dimensions", published_event["payload"])
        # Real embedding is 256-dimensional
        self.assertEqual(published_event["payload"]["dimensions"], 256)

    def test_create_embedding_returns_256_dimensions(self):
        """Test that _create_embedding always returns a 256-element list of floats."""
        annotations = [{"label": "cat", "confidence": 0.9}]

        # Patch PIL Image.open so no real file is needed
        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.tobytes.return_value = b"\x00" * 1000

        with patch("PIL.Image.open", return_value=mock_img):
            embedding = self.service._create_embedding("/test.jpg", annotations)

        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 256)
        self.assertTrue(all(isinstance(x, float) for x in embedding))

    def test_create_embedding_empty_annotations(self):
        """Test embedding creation with empty annotations still returns 256 floats."""
        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.tobytes.return_value = b"\xab" * 1000

        with patch("PIL.Image.open", return_value=mock_img):
            embedding = self.service._create_embedding("/test.jpg", [])

        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 256)

    def test_create_embedding_fallback_on_error(self):
        """Test that _create_embedding returns a 256-element fallback when image load fails."""
        with patch("PIL.Image.open", side_effect=FileNotFoundError("no file")):
            embedding = self.service._create_embedding("/nonexistent.jpg", [])

        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 256)

    def test_storage_and_retrieval(self):
        """Test that embeddings are stored and can be retrieved after handling an event."""
        image_path = os.path.join("images", "test.jpg")
        annotations = [{"label": "dog", "confidence": 0.8}]
        event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": annotations,
            "stored_at": image_path,
        })

        fake_embedding = [0.5] * 256
        with patch.object(self.service, "_create_embedding", return_value=fake_embedding):
            self.service._handle_annotation_stored(event)

        retrieved = self.service.get_embedding(image_path)
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved, list)
        self.assertEqual(len(retrieved), 256)

    def test_get_embedding_nonexistent(self):
        """Test retrieving an embedding for a non-existent image returns None."""
        retrieved = self.service.get_embedding("/nonexistent.jpg")
        self.assertIsNone(retrieved)

    def test_search_similar_empty_index(self):
        """Test that search_similar returns empty list when index has no entries."""
        results = self.service.search_similar([0.1] * 256, k=5)
        self.assertEqual(results, [])

    def test_search_similar_returns_results(self):
        """Test that search_similar returns nearest neighbours after embeddings are added."""
        fake_embedding = [0.1] * 256
        with patch.object(self.service, "_create_embedding", return_value=fake_embedding):
            image_path = "/img/a.jpg"
            event = create_event(ANNOTATION_STORED, {
                "image_path": image_path,
                "annotations": [],
                "stored_at": image_path,
            })
            self.service._handle_annotation_stored(event)

        results = self.service.search_similar(fake_embedding, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], image_path)
        self.assertIsInstance(results[0][1], float)

    def test_get_total_embeddings(self):
        """Test that get_total_embeddings reflects the number of indexed embeddings."""
        self.assertEqual(self.service.get_total_embeddings(), 0)

        fake_embedding = [0.2] * 256
        with patch.object(self.service, "_create_embedding", return_value=fake_embedding):
            for i in range(3):
                path = f"/img/{i}.jpg"
                event = create_event(ANNOTATION_STORED, {
                    "image_path": path,
                    "annotations": [],
                    "stored_at": path,
                })
                self.service._handle_annotation_stored(event)

        self.assertEqual(self.service.get_total_embeddings(), 3)


if __name__ == "__main__":
    unittest.main()
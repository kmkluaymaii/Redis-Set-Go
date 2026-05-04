import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# faiss is a compiled C extension absent from CI.  Stub it in sys.modules
# before any service module is imported so `import faiss` never fails.
# ---------------------------------------------------------------------------
_mock_faiss = MagicMock()
sys.modules.setdefault("faiss", _mock_faiss)

import os
import numpy as np


def _make_faiss_index():
    """Return a MagicMock FAISS index that correctly tracks ntotal."""
    index = MagicMock()
    _store = []

    def _add(vec):
        _store.append(vec)
        index.ntotal = len(_store)

    def _search(query, k):
        n = min(k, len(_store))
        distances = [([0.0] * n)]
        indices = [list(range(n))]
        return distances, indices

    index.add.side_effect = _add
    index.search.side_effect = _search
    index.ntotal = 0
    return index


# ---------------------------------------------------------------------------
# Patch MongoClient in *its own module* so that DocumentDBService.__init__
# (which is imported locally inside _load_existing_embeddings) never tries
# to open a real TCP connection.  This patch stays active for the whole
# test module.
# ---------------------------------------------------------------------------
_mongo_patcher = patch("services.document_db.MongoClient", autospec=True)
_mock_mongo = _mongo_patcher.start()
_mock_mongo.return_value.__getitem__.return_value.__getitem__.return_value = MagicMock()

from services.embedding import EmbeddingService  # noqa: E402  (import after stubs)
from messaging.events import ANNOTATION_STORED, EMBEDDING_CREATED, create_event  # noqa: E402


class TestEmbeddingService(unittest.TestCase):

    def setUp(self):
        self.mock_broker = MagicMock()
        self.mock_index = _make_faiss_index()
        _mock_faiss.IndexFlatL2.return_value = self.mock_index

        # EmbeddingService.__init__ calls faiss.IndexFlatL2 — that's fine now.
        # start() calls _load_existing_embeddings, which internally instantiates
        # DocumentDBService — that's fine because MongoClient is patched above.
        self.service = EmbeddingService(self.mock_broker)
        # Replace whatever index the constructor built with our trackable mock.
        self.service.index = self.mock_index

    @classmethod
    def tearDownClass(cls):
        _mongo_patcher.stop()

    # ------------------------------------------------------------------

    def test_start_subscribes_to_topic(self):
        """start() must subscribe to annotation.stored."""
        self.service.start()
        self.mock_broker.subscribe.assert_called_once_with(
            "annotation.stored", self.service._handle_annotation_stored
        )

    def test_handle_annotation_stored_publishes_event(self):
        """annotation.stored events must trigger an embedding.created publish."""
        image_path = "/path/to/test/image.jpg"
        annotations = [{"label": "object", "confidence": 0.9}]
        event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": annotations,
            "stored_at": image_path,
        })

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
        self.assertEqual(published_event["payload"]["dimensions"], 256)

    def test_create_embedding_returns_256_dimensions(self):
        """_create_embedding must return a 256-element list of floats."""
        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.tobytes.return_value = b"\x00" * 1000

        with patch("PIL.Image.open", return_value=mock_img):
            embedding = self.service._create_embedding("/test.jpg", [{"label": "cat", "confidence": 0.9}])

        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 256)
        self.assertTrue(all(isinstance(x, float) for x in embedding))

    def test_create_embedding_empty_annotations(self):
        """_create_embedding with no annotations must still return 256 floats."""
        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.tobytes.return_value = b"\xab" * 1000

        with patch("PIL.Image.open", return_value=mock_img):
            embedding = self.service._create_embedding("/test.jpg", [])

        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 256)

    def test_create_embedding_fallback_on_error(self):
        """_create_embedding must return a 256-element fallback when the image cannot be loaded."""
        with patch("PIL.Image.open", side_effect=FileNotFoundError("no file")):
            embedding = self.service._create_embedding("/nonexistent.jpg", [])

        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 256)

    def test_storage_and_retrieval(self):
        """Embeddings must be stored in the dict and retrievable after handling an event."""
        image_path = os.path.join("images", "test.jpg")
        event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": [{"label": "dog", "confidence": 0.8}],
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
        """Retrieving an embedding for an unknown image must return None."""
        self.assertIsNone(self.service.get_embedding("/nonexistent.jpg"))

    def test_search_similar_empty_index(self):
        """search_similar must return an empty list when the FAISS index is empty."""
        self.service.index.ntotal = 0
        self.assertEqual(self.service.search_similar([0.1] * 256, k=5), [])

    def test_search_similar_returns_results(self):
        """search_similar must return nearest neighbours after at least one embedding is indexed."""
        image_path = "/img/a.jpg"
        fake_embedding = [0.1] * 256
        event = create_event(ANNOTATION_STORED, {
            "image_path": image_path,
            "annotations": [],
            "stored_at": image_path,
        })
        with patch.object(self.service, "_create_embedding", return_value=fake_embedding):
            self.service._handle_annotation_stored(event)

        results = self.service.search_similar(fake_embedding, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], image_path)
        self.assertIsInstance(results[0][1], float)

    def test_get_total_embeddings(self):
        """get_total_embeddings must reflect the FAISS index size."""
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
import os
import tempfile
import unittest
from unittest.mock import MagicMock
from services.cli import CLIService
from messaging.events import QUERY_SUBMITTED

class TestCLIService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        self.temp_dir = tempfile.mkdtemp()
        self.cli = CLIService(self.mock_broker, upload_dir=self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upload_image_delegates_to_upload_service(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jpg') as f:
            f.write('test image content')
            temp_file_path = f.name

        try:
            result_path = self.cli.upload_image(temp_file_path)
            self.assertTrue(os.path.exists(result_path))
            self.assertEqual(os.path.dirname(result_path), self.temp_dir)
            self.mock_broker.publish.assert_called_once()
            self.assertEqual(self.mock_broker.publish.call_args[0][0], 'image.submitted')
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_submit_query_publishes_query_event(self):
        query_text = 'Find annotations for cat'
        event = self.cli.submit_query(query_text)

        self.mock_broker.publish.assert_called_once()
        self.assertEqual(self.mock_broker.publish.call_args[0][0], QUERY_SUBMITTED)
        published_event = self.mock_broker.publish.call_args[0][1]
        self.assertEqual(published_event['payload']['query'], query_text)
        self.assertEqual(event, published_event)

if __name__ == '__main__':
    unittest.main()

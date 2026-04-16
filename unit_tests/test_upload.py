import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from services.upload import UploadService
from messaging.events import IMAGE_SUBMITTED

class TestUploadService(unittest.TestCase):
    def setUp(self):
        self.mock_broker = MagicMock()
        # Use a temporary directory for uploads
        self.temp_dir = tempfile.mkdtemp()
        self.service = UploadService(self.mock_broker, upload_dir=self.temp_dir)

    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upload_image_success(self):
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jpg') as f:
            f.write("test image content")
            temp_file_path = f.name

        try:
            # Upload the image
            result_path = self.service.upload_image(temp_file_path)
            
            # Verify the file was copied
            self.assertTrue(os.path.exists(result_path))
            self.assertEqual(os.path.dirname(result_path), self.temp_dir)
            
            # Verify the content was copied correctly
            with open(result_path, 'r') as f:
                self.assertEqual(f.read(), "test image content")
            
            # Verify the broker was called
            self.mock_broker.publish.assert_called_once()
            args = self.mock_broker.publish.call_args
            self.assertEqual(args[0][0], IMAGE_SUBMITTED)  # topic
            
            # Check the event payload
            event = args[0][1]
            self.assertEqual(event["topic"], IMAGE_SUBMITTED)
            self.assertEqual(event["payload"]["original_path"], temp_file_path)
            self.assertEqual(event["payload"]["stored_path"], result_path)
            self.assertEqual(event["payload"]["filename"], os.path.basename(temp_file_path))
            
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_upload_image_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.service.upload_image("nonexistent_file.jpg")

    def test_upload_dir_creation(self):
        # Test that upload directory is created if it doesn't exist
        new_dir = os.path.join(self.temp_dir, "new_uploads")
        service = UploadService(self.mock_broker, upload_dir=new_dir)
        self.assertTrue(os.path.exists(new_dir))

if __name__ == '__main__':
    unittest.main()

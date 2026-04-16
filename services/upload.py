import os
import shutil
from messaging.broker import RedisBroker
from messaging.events import create_event, IMAGE_SUBMITTED

class UploadService:
    def __init__(self, broker: RedisBroker, upload_dir: str = "uploads"):
        self.broker = broker
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    def upload_image(self, file_path: str) -> str:
        """
        Uploads an image file by copying it to the upload directory
        and publishing an IMAGE_SUBMITTED event.
        
        Args:
            file_path: Path to the image file to upload
            
        Returns:
            The path where the file was stored
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.upload_dir, filename)
        
        # Copy the file to upload directory
        shutil.copy2(file_path, dest_path)
        
        # Create and publish event
        event = create_event(IMAGE_SUBMITTED, {
            "original_path": file_path,
            "stored_path": dest_path,
            "filename": filename
        })
        
        self.broker.publish(IMAGE_SUBMITTED, event)
        
        return dest_path

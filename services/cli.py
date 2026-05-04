from services.upload import UploadService
from messaging.events import create_event, QUERY_SUBMITTED

class CLIService:
    def __init__(self, broker, upload_dir="uploads", image_base="images"):
        self.broker = broker
        self.upload_service = UploadService(self.broker, upload_dir=upload_dir)
        self.image_base = image_base

    def upload_image(self, file_path: str, category: str = None) -> str:
        """Upload an image and publish the image submitted event."""
        return self.upload_service.upload_image(file_path)

    def submit_query(self, query_text: str) -> dict:
        """Publish a query submitted event."""
        event = create_event(QUERY_SUBMITTED, {"query": query_text})
        self.broker.publish(QUERY_SUBMITTED, event)
        return event

    def list_available_images(self, category: str = None):
        """List available images in the image base."""
        import os
        base_path = self.image_base
        if category:
            base_path = os.path.join(self.image_base, category)
        
        if not os.path.exists(base_path):
            return []
        
        images = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    rel_path = os.path.relpath(os.path.join(root, file), self.image_base)
                    images.append(rel_path)
        return images
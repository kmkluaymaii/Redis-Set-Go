from services.upload import UploadService
from messaging.events import create_event, QUERY_SUBMITTED

class CLIService:
    def __init__(self, broker, upload_dir="uploads"):
        self.broker = broker
        self.upload_service = UploadService(self.broker, upload_dir=upload_dir)

    def upload_image(self, file_path: str) -> str:
        """Upload an image and publish the image submitted event."""
        return self.upload_service.upload_image(file_path)

    def submit_query(self, query_text: str) -> dict:
        """Publish a query submitted event."""
        event = create_event(QUERY_SUBMITTED, {"query": query_text})
        self.broker.publish(QUERY_SUBMITTED, event)
        return event
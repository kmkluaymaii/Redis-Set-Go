from messaging.broker import RedisBroker
from messaging.events import INFERENCE_COMPLETED, create_event

class InferenceService:
    def __init__(self, broker: RedisBroker):
        self.broker = broker
        self.broker.subscribe("image.submitted", self._handle_image_submitted)

    def _handle_image_submitted(self, event):
        """Handle an image submission event by running inference."""
        image_path = event["payload"]["stored_path"]
        
        # Generate basic inference results
        annotations = self._run_inference(image_path)

        # Create and publish completion event
        result_event = create_event(INFERENCE_COMPLETED, {
            "image_path": image_path,
            "annotations": annotations
        })

        self.broker.publish(INFERENCE_COMPLETED, result_event)

    def _run_inference(self, image_path: str) -> list:
        """Run basic inference on the image."""
        return [{"label": "object", "confidence": 0.9}]

import json
import time
from messaging.broker import RedisBroker
from messaging.events import safe_handle, INFERENCE_COMPLETED, create_event

class InferenceService:
    def __init__(self, broker: RedisBroker):
        self.broker = broker
        self.running = False

    def start(self):
        """Start the inference service by subscribing to image submission events."""
        self.running = True
        print("[Inference] Starting inference service...")
        self.broker.subscribe("image.submitted", self._handle_image_submitted)

    def stop(self):
        """Stop the inference service."""
        self.running = False
        print("[Inference] Stopping inference service...")

    def _handle_image_submitted(self, event):
        """Handle an image submission event by running inference."""
        if not self.running:
            return

        try:
            image_path = event["payload"]["stored_path"]
            print(f"[Inference] Processing image: {image_path}")

            # Simulate inference processing time
            time.sleep(0.1)

            # Generate mock inference results
            annotations = self._run_inference(image_path)

            # Create and publish completion event
            result_event = create_event(INFERENCE_COMPLETED, {
                "image_path": image_path,
                "annotations": annotations,
                "confidence": 0.95,
                "model_version": "v1.0"
            })

            self.broker.publish(INFERENCE_COMPLETED, result_event)
            print(f"[Inference] Completed inference for {image_path}")

        except Exception as e:
            print(f"[Inference] Error processing image: {e}")

    def _run_inference(self, image_path: str) -> list:
        """
        Run inference on the image.
        For now, returns mock annotations. In a real implementation,
        this would load an ML model and process the image.
        """
        # Mock inference results
        return [
            {
                "label": "cat",
                "confidence": 0.92,
                "bbox": [0.1, 0.2, 0.8, 0.9]
            },
            {
                "label": "dog",
                "confidence": 0.87,
                "bbox": [0.3, 0.1, 0.7, 0.8]
            }
        ]

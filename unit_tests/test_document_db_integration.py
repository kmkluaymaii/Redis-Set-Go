#!/usr/bin/env python3

import sys
import os
import threading
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from messaging.broker import RedisBroker
from services.document_db import DocumentDBService
from messaging.events import create_event, INFERENCE_COMPLETED

def test_document_db():
    broker = RedisBroker()

    # Start DB service in a thread
    def run_db_service():
        db_service = DocumentDBService(broker)
        db_service.start()

    db_thread = threading.Thread(target=run_db_service, daemon=True)
    db_thread.start()

    time.sleep(1)  # Wait for service to start

    # Simulate inference completed event
    event = create_event(INFERENCE_COMPLETED, {
        "image_path": "/uploads/test.jpg",
        "annotations": [{"label": "cat", "confidence": 0.95}]
    })

    # Publish the event
    broker.publish(INFERENCE_COMPLETED, event)
    print("Event published")

    # Wait for processing
    time.sleep(2)
    print("Waited for processing")

    # Create a service instance just for querying (don't start it)
    query_service = DocumentDBService(broker)  # Don't call start()

    # Retrieve annotations
    annotations = query_service.get_annotations("/uploads/test.jpg")
    print(f"Retrieved annotations: {annotations}")

    if annotations == [{"label": "cat", "confidence": 0.95}]:
        print("Document DB with MongoDB is working!")
        return True
    else:
        print("Document DB failed.")
        return False

if __name__ == "__main__":
    success = test_document_db()
    sys.exit(0 if success else 1)
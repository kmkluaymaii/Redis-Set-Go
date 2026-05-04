#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from messaging.broker import RedisBroker
from messaging.events import create_event, safe_handle
import time
import threading

def test_pubsub():
    broker = RedisBroker()

    # Test event
    test_event = create_event("test.topic", {"message": "Hello, pub/sub!"})

    received_events = []

    def handler(event):
        print(f"Received event: {event}")
        received_events.append(event)

    # Subscribe in a thread
    def subscriber():
        broker.subscribe("test.topic", handler)

    sub_thread = threading.Thread(target=subscriber, daemon=True)
    sub_thread.start()

    # Wait a bit for subscription
    time.sleep(1)

    # Publish
    broker.publish("test.topic", test_event)

    # Wait for message
    time.sleep(2)

    if received_events:
        print("Pub/sub is working! Event received.")
        return True
    else:
        print("Pub/sub failed. No event received.")
        return False

if __name__ == "__main__":
    success = test_pubsub()
    sys.exit(0 if success else 1)
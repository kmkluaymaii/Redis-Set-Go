import uuid
from datetime import datetime

# Create a simple event structure
def create_event(topic, payload):
    return {
        "type": "publish",
        "topic": topic,
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "payload": payload,
        "timestamp": datetime.utcnow().isoformat()
    }

# Validation whether the event has all required fields and correct format
def validate_event(event):
    required_fields = ["type", "topic", "event_id", "payload", "timestamp"]
    return all(field in event for field in required_fields)

# Track if an event has already been processed to avoid duplicates
processed_events = set()
def is_duplicate(event):
    event_id = event.get("event_id")

    if event_id in processed_events:
        return True

    processed_events.add(event_id)
    return False

# Safely handle an event with validation and duplicate checking
def safe_handle(event, handler):
    if not validate_event(event):
        print("[Event] Invalid event, skipping")
        return

    if is_duplicate(event):
        print("[Event] Duplicate event, skipping")
        return

    try:
        handler(event)
    except Exception as e:
        print(f"[Event] Error: {e}")

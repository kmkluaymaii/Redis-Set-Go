import redis
import json

class RedisBroker:
    def __init__(self, host="localhost", port=6379):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)

    def publish(self, topic, event):
        self.redis.publish(topic, json.dumps(event))

    def subscribe(self, topic, handler):
        pubsub = self.redis.pubsub()
        pubsub.subscribe(topic)

        print(f"Subscribed to {topic}")

        for message in pubsub.listen():
            if message["type"] != "message":
                continue

            try:
                event = json.loads(message["data"])
                handler(event)

            except Exception as e:
                print(f"[Broker] Error processing message: {e}")
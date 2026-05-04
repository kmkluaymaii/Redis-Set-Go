from messaging.broker import RedisBroker
from messaging.events import INFERENCE_COMPLETED, create_event
from PIL import Image
import random
import os

class InferenceService:
    def __init__(self, broker: RedisBroker):
        self.broker = broker

    def start(self):
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
        """Run simulated inference on the image with animal-specific detections."""
        try:
            # Load image to get properties
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Extract animal name from filename
                filename = os.path.basename(image_path).lower()
                animal_name = filename.split('.')[0]  # Remove extension
                
                # Animal-specific detection characteristics
                animal_features = {
                    'dog': ['dog', 'animal', 'pet', 'mammal'],
                    'cat': ['cat', 'animal', 'pet', 'feline'],
                    'rabbit': ['rabbit', 'animal', 'mammal', 'bunny'],
                    'deer': ['deer', 'animal', 'mammal', 'wildlife'],
                    'giraffe': ['giraffe', 'animal', 'mammal', 'wildlife']
                }
                
                # Get features for this animal, fallback to generic animal
                features = animal_features.get(animal_name, ['animal', 'mammal', 'wildlife'])
                
                # Generate 2-4 feature detections
                num_features = min(len(features), random.randint(2, 4))
                selected_features = random.sample(features, num_features)
                
                annotations = []
                for i, feature in enumerate(selected_features):
                    # Random bounding box within image (larger for main animal)
                    if feature == animal_name:  # Main animal gets larger bbox
                        x1 = random.randint(0, width // 4)
                        y1 = random.randint(0, height // 4)
                        x2 = random.randint(x1 + width//2, width)
                        y2 = random.randint(y1 + height//2, height)
                        confidence = round(random.uniform(0.90, 0.98), 2)
                    else:  # Other features get smaller bboxes
                        x1 = random.randint(0, width // 3)
                        y1 = random.randint(0, height // 3)
                        x2 = random.randint(x1 + width//6, width)
                        y2 = random.randint(y1 + height//6, height)
                        confidence = round(random.uniform(0.75, 0.92), 2)
                    
                    annotations.append({
                        "label": feature.replace('_', ' ').title(),
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "animal": animal_name.title()
                    })
                
                # Add some generic environmental objects
                generic_objects = ['grass', 'tree', 'sky', 'ground', 'background']
                for _ in range(random.randint(1, 2)):
                    x1 = random.randint(0, width // 2)
                    y1 = random.randint(0, height // 2)
                    x2 = random.randint(x1 + 20, width)
                    y2 = random.randint(y1 + 20, height)
                    
                    label = random.choice(generic_objects)
                    confidence = round(random.uniform(0.6, 0.8), 2)
                    
                    annotations.append({
                        "label": label,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
                
                return annotations
                
        except Exception as e:
            print(f"Inference error: {e}")
            return [{"label": "landmark", "confidence": 0.9, "bbox": [0, 0, 100, 100]}]

#!/usr/bin/env python3

import sys
import os
import time
sys.path.append(os.path.dirname(__file__))

from messaging.broker import RedisBroker
from services.upload import UploadService
from services.inference import InferenceService
from services.document_db import DocumentDBService
from services.embedding import EmbeddingService
from services.query import QueryService
from services.cli import CLIService

def process_all_images():
    """Process all images in the images folder through the full pipeline."""

    print("Starting Redis-Set-Go Image Processing Pipeline")
    print("=" * 50)

    # Initialize services
    broker = RedisBroker()
    upload_service = UploadService(broker)

    # Start services in threads
    def start_inference():
        inference_service = InferenceService(broker)
        inference_service.start()

    def start_db():
        db_service = DocumentDBService(broker)
        db_service.start()

    def start_embedding():
        embedding_service = EmbeddingService(broker)
        embedding_service.start()

    def start_query():
        query_service = QueryService(broker)
        query_service.start()

    # Start all services
    import threading
    threads = []
    for func in [start_inference, start_db, start_embedding, start_query]:
        t = threading.Thread(target=func, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.1)  # Small delay between starts

    # Wait a bit for services to be ready
    time.sleep(1)

    # Create shared embedding service for all processing
    shared_embedding_service = EmbeddingService(broker)

    # Get all images
    image_files = [f for f in os.listdir("images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📁 Found {len(image_files)} images: {image_files}")

    processed_images = []

    for image_file in image_files:
        image_path = os.path.join("images", image_file)
        animal_name = os.path.splitext(image_file)[0]  # Remove extension

        print(f"\n🐾  Processing {animal_name.upper()}...")

        try:
            # Upload image
            uploaded_path = upload_service.upload_image(image_path)
            print(f"Uploaded: {uploaded_path}")

            # For testing: run inference synchronously to see immediate results
            inference_service = InferenceService(broker)
            # Don't call start() to avoid blocking
            annotations = inference_service._run_inference(uploaded_path)
            
            # Store annotations directly (without starting subscription)
            db_check = DocumentDBService(broker)
            # Manually store without subscription
            db_check.collection.insert_one({
                "image_path": uploaded_path,
                "annotations": annotations,
                "timestamp": time.time()
            })
            
            print(f"🔍 Detected {len(annotations)} objects")
            for ann in annotations[:3]:  # Show first 3
                bbox = ann.get('bbox', [])
                label = ann.get('label', 'unknown')
                confidence = ann.get('confidence', 0.0)
                animal = ann.get('animal', '')
                if animal:
                    print(f"   - {label} (confidence: {confidence:.2f}) in {animal}")
                else:
                    print(f"   - {label} (confidence: {confidence:.2f}) at {bbox}")

            # Generate embedding
            # Use the shared embedding service
            embedding = shared_embedding_service._create_embedding(uploaded_path, annotations)
            
            if embedding:
                print(f"Generated {len(embedding)}-dimensional embedding")
                # Store embedding in the shared service's dictionary and FAISS index
                shared_embedding_service.embeddings[uploaded_path] = embedding
                # Add to FAISS index
                import numpy as np
                embedding_np = np.array([embedding], dtype=np.float32)
                shared_embedding_service.index.add(embedding_np)
                shared_embedding_service.image_paths.append(uploaded_path)
            else:
                print("No embedding generated")

            processed_images.append({
                'animal': animal_name,
                'path': uploaded_path,
                'annotations': annotations,
                'embedding': embedding
            })

        except Exception as e:
            print(f"Error processing {animal_name}: {e}")

    print(f"\nProcessed {len(processed_images)}/{len(image_files)} images successfully!")

    # Test querying - create a new query service that shares the embedding store
    print("\nTesting queries...")
    embedding_service = shared_embedding_service  # Use the shared instance
    query_service = QueryService(broker, embedding_service)
    test_queries = ["dog", "cat", "rabbit", "deer", "giraffe"]

    for query in test_queries:
        results = query_service.get_results_for_query(query)
        print(f"Query '{query}': found {len(results)} matches")

    print("\nPipeline Summary:")
    print(f"   Images processed: {len(processed_images)}")
    print(f"   Total annotations: {sum(len(img['annotations']) for img in processed_images)}")
    print(f"   Embeddings created: {sum(1 for img in processed_images if img['embedding'])}")

    return processed_images

if __name__ == "__main__":
    results = process_all_images()
    print("\nProcessing complete! You can now query your image database.")
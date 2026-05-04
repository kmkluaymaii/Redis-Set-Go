# Redis Set Go!

Project by Rawisara Chairat (rawisara@bu.edu) and Pippi Pi (ppp@bu.edu)

### System Overview
The system is composed by services connected through an event-driven pipeline:

```Image Upload → Inference → Document DB → Embeddings → FAISS → Query```

### Defined Services
**1. Upload Service** (upload.py): Handles image ingestion and triigers processing events.

**2. Inference Service** (inference.py): Performs animal detection and generates annotations with confidence scores.

**3. Document DB Service** (document_db.py): Stores and retrieves annotations and embedding metadata (MongoDB-backed).

**4. Embedding Service** (embedding.py): Generates 256-dimensional vector embeddings for images and metadata.

**5. Query Service** (query.py): Performs similarity search using FAISS to retrieve related images.

**6. CLI Service** (cli.py): Provides a command-line interface to interact with the system.

### Sample Run

Below is an example execution of the full pipeline processing 5 animals images:

Run this command: 
```
python process_images.py
```
Result:
```
Starting Redis-Set-Go Image Processing Pipeline
==================================================
Subscribed to image.submitted
Subscribed to inference.completed
Subscribed to annotation.stored
Subscribed to query.submitted
📁 Found 5 images: ['dog.jpg', 'rabbit.jpg', 'deer.jpg', 'cat.jpg', 'giraffe.jpg']

🐾  Processing DOG...
Uploaded: uploads/dog.jpg
🔍 Detected 3 objects
   - Mammal (confidence: 0.82) in Dog
   - Pet (confidence: 0.78) in Dog
   - background (confidence: 0.68) at [135, 184, 544, 574]
Generated 256-dimensional embedding

🐾  Processing RABBIT...
Uploaded: uploads/rabbit.jpg
🔍 Detected 4 objects
   - Mammal (confidence: 0.79) in Rabbit
   - Bunny (confidence: 0.87) in Rabbit
   - Animal (confidence: 0.82) in Rabbit
Generated 256-dimensional embedding

🐾  Processing DEER...
Uploaded: uploads/deer.jpg
🔍 Detected 5 objects
   - Mammal (confidence: 0.78) in Deer
   - Wildlife (confidence: 0.91) in Deer
   - Deer (confidence: 0.90) in Deer
Generated 256-dimensional embedding

🐾  Processing CAT...
Uploaded: uploads/cat.jpg
🔍 Detected 3 objects
   - Cat (confidence: 0.93) in Cat
   - Feline (confidence: 0.80) in Cat
   - tree (confidence: 0.65) at [623, 385, 980, 777]
Generated 256-dimensional embedding

🐾  Processing GIRAFFE...
Uploaded: uploads/giraffe.jpg
🔍 Detected 5 objects
   - Wildlife (confidence: 0.79) in Giraffe
   - Mammal (confidence: 0.89) in Giraffe
   - Animal (confidence: 0.88) in Giraffe
Generated 256-dimensional embedding

Processed 5/5 images successfully!

Testing queries...
Query 'dog': found 5 matches
Query 'cat': found 5 matches
Query 'rabbit': found 5 matches
Query 'deer': found 5 matches
Query 'giraffe': found 5 matches

Pipeline Summary:
   Images processed: 5
   Total annotations: 20
   Embeddings created: 5

Processing complete! You can now query your image database.
```

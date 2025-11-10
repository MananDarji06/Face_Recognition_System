# Face Recognition System
GUI and algos made in Python

## Guide:
Face Recognition Starter (facenet-pytorch) - single-file demo
Filename: face_recog_starter.py

What this file provides:
- Enrollment: select a folder of images per person, extract embeddings (facenet-pytorch MTCNN + InceptionResnetV1), and store into a local SQLite DB.
- Recognition: select an image (or use webcam) -> detect faces, compute embeddings, match to enrolled people using cosine distance, display results in a Tkinter GUI with image preview and bounding boxes.
- History: prediction records saved into SQLite and exportable to CSV.
- Visualization: helper to plot genuine vs impostor distance histograms using Seaborn.

Notes:
- facenet-pytorch will download pretrained weights the first time it runs (internet required initially).
- This starter is designed for demo/uni use and not production. It stores embeddings as pickled blobs in SQLite.
- Ensure you run this file from the project root where a "data/" directory can be created.

Usage:
1. python face_recog_starter.py
2. Use the GUI buttons: Enroll Person, Predict Image, View History, Export History, Plot Distances.

## Function Logic:
algorithms.py (minimal algorithms)

Globals

DEVICE — 'cuda' if a GPU is available, else 'cpu'.

DEFAULT_THRESHOLD — cosine-distance cutoff for deciding Unknown (default 0.6).

class FaceEngine

__init__(self, device: str | None = None)
Sets the compute device, loads MTCNN (face detect+align) and InceptionResnetV1 (512-D face embeddings, pretrained on VGGFace2). No DB/GUI here.

image_to_pil(img_or_path) → PIL.Image
Accepts a file path or an OpenCV BGR ndarray. Returns an RGB PIL Image the models can use.

embed_faces(self, pil_img) → ([np.ndarray], boxes | None)

Runs MTCNN to align faces (returns Tensor[n,3,160,160]).

Passes aligned faces through the embedder to get one 512-D vector per face (as numpy arrays).

Gets face bounding boxes as (x1,y1,x2,y2); returns (embeddings_list, boxes_or_None).

Module functions

_cosine_distance(query_vec, gallery) → np.ndarray
Computes 1 − cosine_similarity between a single query vector and each vector in a gallery ((N,) distances). Lower = more similar.

match_embedding(query_emb, known, threshold=DEFAULT_THRESHOLD) → (label, best_dist)
Nearest-neighbor matcher.
known is [(name, emb), ...]. Finds the closest name. If best_dist <= threshold → that name, else "Unknown".

draw_boxes_on_pil(pil_img, boxes, labels=None) → PIL.Image
Draws green rectangles for each box. If labels like [(name, distance), ...] provided, writes “name (0.xx)” above the box. Returns a new PIL image.

gui_app.py (Tkinter GUI)

class FaceRecogGUI

__init__(self, root, repo, engine, threshold=DEFAULT_THRESHOLD)
Wires up the UI: buttons (Enroll Folder/Webcam, Predict Image, Live Predict, History, Export), threshold controls, image preview area.

repo is your DB layer (from main.py), and must implement:
insert_person, insert_embedding, fetch_all_embeddings, insert_prediction, fetch_predictions.

_set_threshold(self)
Reads the threshold from the entry box, validates, and updates self.threshold.

enroll_person_dialog(self)
Folder-based enrollment (for a person who has a folder of images).

Ask for folder + person name.

In a background thread, for each image: extract embeddings via engine.embed_faces(...).

Save each embedding with repo.insert_embedding(...).

Notify with a message box when done.

enroll_via_webcam(self)
Webcam-based enrollment (no dataset needed).

Ask for person name and number of frames to collect (e.g., 20).

Open camera; every few frames, take the first detected face → embedding → repo.insert_embedding(...).

Shows a live OpenCV window with “Collected: X/N”. Stop with Q or Esc.

Message when complete.

predict_image_dialog(self)
Single-image prediction.

Pick an image.

Compute embeddings & boxes.

Load known = repo.fetch_all_embeddings().

For each face, match_embedding(...) → (label, dist); log via repo.insert_prediction(...).

Draw boxes/labels on the image, show in the GUI.

live_predict_webcam(self)
Real-time recognition from the camera.

Open webcam, for each frame detect faces, embed, load known, match, and draw results directly on the OpenCV frame for speed.

Stop with Q / Esc. (Runs in a background thread.)

view_history(self)
Fetches up to 200 prediction rows via repo.fetch_predictions(...) and displays them in a simple Treeview table.

export_history(self)
Fetches up to 10k predictions and saves to a CSV chosen by the user.

Helper

simple_input_dialog(parent, prompt) → str | None
Tiny modal dialog to capture a single text input (used for person name, frame count, etc.).

main.py (MongoDB + app entry)

MongoRepo.__init__(self, uri, db_name):
Creates a MongoDB client using the given URI, selects the database,
defines collection handles (people, embeddings, predictions),
and ensures required indexes exist.


MongoRepo._ensure_indexes(self):
Creates indexes for efficient lookups:
- people.name is unique (no duplicate person names)
- embeddings.person_id for fast join-like queries
- predictions.created_at for sorting by time


MongoRepo.insert_person(self, name):
Inserts a new person document if it doesn't exist.
Uses upsert=True so existing names are not duplicated.
Returns the ObjectId of the person.


MongoRepo.insert_embedding(self, person_id, embedding, source_image):
Stores a face embedding for a person.
Embedding (numpy array) is pickled and wrapped in Binary so MongoDB can store it.
Also saves the source image path or 'webcam_frame'.


MongoRepo.fetch_all_embeddings(self):
Performs an aggregation pipeline that joins embeddings with the people collection,
so each embedding is returned with the correct person name.
Unpickles each embedding blob back into a numpy array.
Returns a list of tuples: (person_name, embedding_vector).


MongoRepo.insert_prediction(self, image_path, predicted_person, distance):
Stores the result of a prediction: image used, predicted name,
the cosine distance, model used, and timestamp.


MongoRepo.fetch_predictions(self, limit):
Queries the predictions collection, sorted by most recent.
Converts results to a pandas DataFrame for easy display/export in GUI.


main():
Creates a MongoRepo instance, initializes FaceEngine, builds the Tkinter window,
and starts the event loop. This is the entry point of the application.

## How to use:
- Activate the virtual environment
(Windows: venv\Scripts\activate, masOS/Linux:source venv/bin/activate)
- Install required dependencies (Only needed the first time)
(pip install -r requirements.txt)
- Run the "main.py" file in the terminal
(python main.py)
- The main GUI window will open
![alt text](image.png)
- To enroll a new person using the webcam
Click “Enroll (Webcam)"
Enter the person’s name
Follow the voice instructions:
Show Front Profile → 5 frames captured
Show Left Profile → 5 frames captured
Show Right Profile → 5 frames captured
Wait until you see the message confirming successful enrollment.
- To enroll a person from existing images
Click “Enroll (Images)”
Select one or more .jpg/.jpeg/.png files
Enter the person’s name
The system extracts face embeddings and stores them in MongoDB.
- To perform face recognition on a single image
Click “Predict Image”
Select an image file
The system detects faces, compares embeddings, and shows predicted identities.
- To use real-time face recognition
Click “Live Predict”
The webcam feed opens
Detected faces are labeled with predicted names and distances
Press Q to exit.
- To view past recognition attempts
Click “View History”
A table opens showing:
Image path
Predicted person
Distance score
Timestamp
- To export the prediction history
Click “Export History”
Choose a location to save the CSV file
A .csv file containing the logged predictions is created.
- To stop or exit the application
Simply close the GUI window or press Ctrl + C in the terminal.
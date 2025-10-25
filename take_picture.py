import cv2
import numpy as np
from deepface import DeepFace
import sqlite3
import uuid
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect('attendance.db')
c = conn.cursor()

# Create table to store student embeddings
c.execute('''CREATE TABLE IF NOT EXISTS embeddings
             (student_id TEXT PRIMARY KEY, embedding BLOB)''')

# Initialize the deep face model
model = DeepFace.build_model('Facenet')

def capture_and_embed():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Display the captured frame
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()
        
        # Detect faces in the frame
        faces = DeepFace.detectFace(frame, detector_backend='opencv')

        # Extract embeddings for each detected face
        for face in faces:
            # Resize face for better embedding extraction
            resized_face = cv2.resize(face, (160, 160))
            embedding = DeepFace.represent(resized_face, model=model)

            # Generate a unique student ID
            student_id = str(uuid.uuid4())

            # Insert embedding and student ID into the database
            c.execute("INSERT OR REPLACE INTO embeddings (student_id, embedding) VALUES (?, ?)", (student_id, embedding.tobytes()))
            conn.commit()

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start capturing and embedding
capture_and_embed()

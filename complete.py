import cv2
import math
import dlib
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os
import shutil

# Load the YOLOR model with face detection configuration
model = YOLO("yolov8s.pt")

# Load the dlib face detector
detector = dlib.get_frontal_face_detector()

# Define thresholds for EAR and MAR
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.2


def faceRecognition(img, cropped_objects_dir):
    # Path to the directory to save unknown faces
    unknown_faces_dir = "./unknown/"

    # Path to the directory to save known faces
    known_faces_dir = "./known/"

    # Initialize a list to store the extracted names
    extracted_names = []

    # Check if the 'unknown' folder exists, otherwise create it
    if not os.path.exists(unknown_faces_dir):
        os.makedirs(unknown_faces_dir)
    else:
        # If the 'unknown' folder exists, clear all files and subfolders
        for file_or_folder in os.listdir(unknown_faces_dir):
            file_or_folder_path = os.path.join(unknown_faces_dir, file_or_folder)
            if os.path.isfile(file_or_folder_path):
                os.remove(file_or_folder_path)
            elif os.path.isdir(file_or_folder_path):
                shutil.rmtree(file_or_folder_path)

    # Check if the 'known' folder exists, otherwise create it
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    else:
        # If the 'known' folder exists, clear all files and subfolders
        for file_or_folder in os.listdir(known_faces_dir):
            file_or_folder_path = os.path.join(known_faces_dir, file_or_folder)
            if os.path.isfile(file_or_folder_path):
                os.remove(file_or_folder_path)
            elif os.path.isdir(file_or_folder_path):
                shutil.rmtree(file_or_folder_path)

    # Iterate through the image files in the directory
    for filename in os.listdir(cropped_objects_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(cropped_objects_dir, filename)

            model = DeepFace.find(
                img_path=img_path,
                db_path="student_faces",
                enforce_detection=False,
                model_name="Facenet512",
            )

            # Check if a face was recognized in the image
            if model and len(model[0]["identity"]) > 0:
                # Extract the name and append it to the list
                name = model[0]["identity"][0].split("/")[0].split("\\")[1]

                # Save the known face into the 'known' folder
                known_faces_path = os.path.join(
                    known_faces_dir, f"{len(extracted_names) + 1}_{name}.jpg"
                )
                shutil.copy(img_path, known_faces_path)

            else:
                # If no face is recognized, set name to 'unknown'
                name = "Unknown"

                # Save the unknown face into the 'unknown' folder
                unknown_faces_path = os.path.join(
                    unknown_faces_dir, f"{len(extracted_names) + 1}.jpg"
                )
                shutil.copy(img_path, unknown_faces_path)

            extracted_names.append(name)

    return extracted_names


# Function to perform face detection, recognition, and awake/sleepy analysis on a video input
def video_detection(path_x):
    video_capture = path_x

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_capture)

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            # Perform face detection using dlib
            faces = detector(img, 1)
            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Calculate EAR and MAR
                left_eye = img[y1:y2, x1 : x1 + x2 - x1]
                right_eye = img[y1:y2, x2 - x1 : x2]

                left_ear_numerator = sum(
                    [
                        (left_eye[i][0] - left_eye[i + 1][0]) ** 2
                        + (left_eye[i][1] - left_eye[i + 1][1]) ** 2
                        for i in range(5)
                    ]
                )
                right_ear_numerator = sum(
                    [
                        (right_eye[i][0] - right_eye[i + 1][0]) ** 2
                        + (right_eye[i][1] - right_eye[i + 1][1]) ** 2
                        for i in range(5)
                    ]
                )
                left_mar_numerator = sum(
                    [
                        (left_eye[i][0] - left_eye[i + 1][0]) ** 2
                        + (left_eye[i][1] - left_eye[i + 1][1]) ** 2
                        for i in range(4)
                    ]
                )
                right_mar_numerator = sum(
                    [
                        (right_eye[i][0] - right_eye[i + 1][0]) ** 2
                        + (right_eye[i][1] - right_eye[i + 1][1]) ** 2
                        for i in range(4)
                    ]
                )

                left_ear_denominator = 2 * (
                    np.power(left_eye[0][0] - left_eye[4][0], 2)
                    + np.power(left_eye[0][1] - left_eye[4][1], 2)
                )
                right_ear_denominator = 2 * (
                    np.power(right_eye[0][0] - right_eye[4][0], 2)
                    + np.power(right_eye[0][1] - right_eye[4][1], 2)
                )
                left_mar_denominator = 2 * (
                    np.power(left_eye[0][0] - left_eye[4][0], 2)
                    + np.power(left_eye[0][1] - left_eye[4][1], 2)
                )
                right_mar_denominator = 2 * (
                    np.power(right_eye[0][0] - right_eye[4][0], 2)
                    + np.power(right_eye[0][1] - right_eye[4][1], 2)
                )

                left_ear = np.divide(
                    left_ear_numerator,
                    left_ear_denominator,
                    where=left_ear_denominator != 0,
                )
                right_ear = np.divide(
                    right_ear_numerator,
                    right_ear_denominator,
                    where=right_ear_denominator != 0,
                )
                left_mar = np.divide(
                    left_mar_numerator,
                    left_mar_denominator,
                    where=left_mar_denominator != 0,
                )
                right_mar = np.divide(
                    right_mar_numerator,
                    right_mar_denominator,
                    where=right_mar_denominator != 0,
                )

                # Analyze EAR and MAR
                if (
                    np.all(left_ear < EAR_THRESHOLD)
                    and np.all(right_ear < EAR_THRESHOLD)
                    and np.all(left_mar < MAR_THRESHOLD)
                    and np.all(right_mar < MAR_THRESHOLD)
                ):
                    status = "Sleepy"
                else:
                    status = "Awake"

                # Crop and save the detected face
                object_image = img[y1:y2, x1:x2]
                cropped_objects_dir = "./cropped_faces/"
                if not os.path.exists(cropped_objects_dir):
                    os.makedirs(cropped_objects_dir)
                object_image_path = os.path.join(
                    cropped_objects_dir,
                    f"face{len(os.listdir(cropped_objects_dir))}.jpg",
                )
                cv2.imwrite(object_image_path, object_image)

                # Perform face recognition
                names = faceRecognition(img, cropped_objects_dir)
                name = names[len(os.listdir(cropped_objects_dir)) - 1]

                # Display the name and status
                cv2.putText(
                    img,
                    f"Name: {name}, Status: {status}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Face Detection", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


video_detection("class_room.mp4")


import os
import shutil
from deepface import DeepFace
from PIL import Image
from ultralytics import YOLO
from typing import List, Tuple
import tempfile
import mysql.connector


def faceDetection(input_image: str) -> int:
    # Load the input image
    image = Image.open(input_image)

    # Perform face detection using YOLO model
    model = YOLO("C:/Users/sanje/OneDrive/Desktop/1_FACIAL_dummy/beta-testing/best.pt")
    results = model.predict(input_image)[0]

    return faceExtraction(input_image, model, results)


def faceExtraction(input_image: str, model, results) -> int:
    # Load the image
    image = Image.open(input_image)
    detected_objects = []

    if hasattr(results, "boxes") and hasattr(results, "names"):
        for box in results.boxes.xyxy:
            object_id = int(box[-1])
            object_name = results.names.get(object_id)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            detected_objects.append((object_name, (x1, y1, x2, y2)))

    # Create the 'faces' directory if it doesn't exist
    faces_dir = "faces"
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    # Debug: Print the number of detected objects
    print(f"Number of detected objects: {len(detected_objects)}")

    # Crop and save each detected object
    for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_objects):
        print(f"Saving face {i}")
        object_image = image.crop((x1, y1, x2, y2))
        object_image.save(os.path.join(faces_dir, f"face{i}.jpg"))

    return 0


def faceRecognition(image):
    # Path to the directory containing cropped objects
    cropped_objects_dir = "./faces/"

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

            # model = DeepFace.find(img_path=img_path, db_path="student_faces", enforce_detection=False, model_name="Facenet512")

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
                # print(model[0]['identity'][0].split('/')[0])

                # Save the known face into the 'known' folder
                known_faces_path = os.path.join(
                    known_faces_dir, f"{len(extracted_names) + 1}_{name}.jpg"
                )
                shutil.copy(img_path, known_faces_path)

            else:
                # If no face is recognized, set name to 'unknown'
                name = 0

                # Save the unknown face into the 'unknown' folder
                unknown_faces_path = os.path.join(
                    unknown_faces_dir, f"{len(extracted_names) + 1}.jpg"
                )
                shutil.copy(img_path, unknown_faces_path)

            extracted_names.append(name)

            # store_known_names(extracted_names)

    # Now, iterate through student faces and perform facial recognition
    student_faces_dir = "./student_faces/"
    for student_image in os.listdir(student_faces_dir):
        if student_image.endswith(".jpg"):
            student_image_path = os.path.join(student_faces_dir, student_image)
            model = DeepFace.find(
                img_path=student_image_path,
                db_path="images",
                enforce_detection=False,
                model_name="Facenet512",
            )

            if model and len(model[0]["identity"]) > 0:
                student_name = model[0]["identity"][0].split("/")[0].split("\\")[1]
                print(f"Match found for {student_name} in {student_image}")
            else:
                print(f"No match found for {student_image}")
    return extracted_names


def main():
    # Prompt user for attendance information

    # Path to the image for facial recognition
    image = r"C:\Users\sanje\OneDrive\Desktop\1_FACIAL_dummy\beta-testing\group_picture_classroom.jpg"

    # Perform face detection and recognition
    faceDetection(image)
    names = faceRecognition(image)
    print(names)



def main():
    # Path to the image for facial recognition
    image = r"C:\Users\sanje\OneDrive\Desktop\1_FACIAL_dummy\beta-testing\group_picture_classroom.jpg"

    # Perform face detection and recognition
    faceDetection(image)
    names = faceRecognition(image)
    print(names)

    remove_folders(["faces"])


def remove_folders(folder_list: List[str]):
    for folder in folder_list:
        folder_path = os.path.join(os.getcwd(), folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder}")
        else:
            print(f"Folder '{folder}' does not exist.")


# Call the main function to start the program
if __name__ == "__main__":
    main()

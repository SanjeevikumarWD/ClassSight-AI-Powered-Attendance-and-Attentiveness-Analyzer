import os
from deepface import DeepFace
import cv2
from PIL import Image
from ultralytics import YOLO
import os


import os


def count_images_in_folder(folder_path):
    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
    ]  # Add more extensions if needed
    image_count = 0

    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            _, ext = os.path.splitext(file_name)
            if ext.lower() in image_extensions:
                image_count += 1

    return image_count


def process_video(video_path):
    # Check if the output directory exists, if not, create it
    if not os.path.exists("frame_picture"):
        os.makedirs("frame_picture")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize frame counter
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Increment the frame counter
        frame_counter += 1

        # Calculate the current time in seconds since the start of the video
        current_time = frame_counter / fps

        # Check if we are at the start of a new second
        if current_time % 1 == 0:
            # Save the frame as an image with sequential numbering
            cv2.imwrite(f"frame_picture/frame_{int(current_time)}.jpg", frame)

    # Release the video capture object
    cap.release()

    # calling all function to calculate percentage
    image = "frame_picture/frame_1.jpg"
    folder_path = "frame_picture"

    no_of_faces = faceDetection(image)
    num_images = count_images_in_folder(folder_path)
    awake_faces_count = count_awake_faces_in_frames()

    calculate_percentage(no_of_faces, num_images, awake_faces_count)
    # print("Number of awake faces = ", awake_faces_count)


def count_awake_faces_in_frames():
    # Directory where the snapshots are stored
    snapshot_dir = "frame_picture"

    # Initialize counter for awake faces
    awake_count = 0

    # Process each image in the directory
    for filename in os.listdir(snapshot_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(snapshot_dir, filename)
            result = DeepFace.analyze(img_path=filepath, actions=["emotion"])

            # Iterate over each face in the result
            for face_result in result:
                # Check if the face does not exhibit a 'sleepy' emotion or if the 'sleepy' emotion is below a certain threshold
                if (
                    "sleepy" not in face_result["emotion"].keys()
                    or face_result["emotion"]["sleepy"] < 0.5
                ):
                    awake_count += 1

    # Return the total count of awake faces
    return awake_count


def faceDetection(input_image: str) -> int:
    # Load the input image
    image = Image.open(input_image)

    # Perform face detection using YOLO model
    model = YOLO("best.pt")
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

    # Display the total number of objects detected in the frame
    # print(f"Total objects detected in the frame: {len(detected_objects)}")

    return len(detected_objects)


def calculate_percentage(no_of_faces, num_image, awake_faces):
    result = (awake_faces / no_of_faces) * (1 / num_image) * 100

    print("Results for attentiveness of the class")
    print("Average attentive percentage: {:.2f}%".format(result))


src = "class_room.mp4"
process_video(src)

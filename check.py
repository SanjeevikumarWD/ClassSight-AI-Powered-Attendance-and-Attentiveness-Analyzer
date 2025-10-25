import cv2
from retinaface import RetinaFace


def detect_faces(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Use RetinaFace for face detection
    faces = RetinaFace.detect_faces(img)

    # Draw rectangles around detected faces
    for face in faces:
        x1, y1, x2, y2 = face["facial_area"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the image with detected faces
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Provide the path to the group picture
image_path = "group_picture_classroom.jpg"

# Call the function to detect faces
detect_faces(image_path)

import os
import cv2
import pyheif
import shutil
import tempfile
from PIL import Image


# Function to detect faces in an image
def detect_faces(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    )

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6, minSize=(100, 100)
    )

    return faces


# Function to convert HEIC image to JPG
def convert_heic_to_jpg(heic_path, jpg_path):
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    image.save(jpg_path, "JPEG")


# Function to process images in the input folder
def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Counter for processed images
    num_processed_images = 0

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".heic")):
            num_processed_images += 1

            input_path = os.path.join(input_folder, filename)

            # Convert HEIC to JPG
            if filename.endswith(".heic"):
                # Create a temporary directory for conversion
                temp_dir = tempfile.mkdtemp()
                temp_jpg_path = os.path.join(
                    temp_dir, os.path.splitext(filename)[0] + ".jpg"
                )
                convert_heic_to_jpg(input_path, temp_jpg_path)
                input_path = temp_jpg_path

            # Read the image
            image = cv2.imread(input_path)

            # Detect faces in the image
            faces = detect_faces(image)

            for i, (x, y, w, h) in enumerate(faces):
                # Crop the face to 512x512
                face_crop = image[y : y + h, x : x + w]
                face_crop = cv2.resize(face_crop, (512, 512))

                # Save the cropped face image to the root of the output directory
                output_face_path = os.path.join(
                    output_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg"
                )
                cv2.imwrite(output_face_path, face_crop)

            # If HEIC file was converted, delete temporary directory
            if filename.endswith(".heic"):
                shutil.rmtree(temp_dir)

    print(f"Processed {num_processed_images} images.")


# Define input and output folders
input_folder = "input"
output_folder = "output"

# Process images
process_images(input_folder, output_folder)

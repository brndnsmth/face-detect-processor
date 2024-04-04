import os
import cv2
import shutil
import tempfile
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


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


# Function to convert HEIC image to JPG using pillow-heif
def convert_heic_to_jpg(heic_path, jpg_path):
    try:
        image = Image.open(heic_path)
        image = image.convert("RGB")
        image.save(jpg_path, format="JPEG")
        return True
    except Exception as e:
        print(f"Error converting HEIC to JPG: {e}")
        return False


# Function to convert image to JPEG format
def convert_to_jpg(image_path, jpg_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to read image")
        cv2.imwrite(jpg_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return True
    except Exception as e:
        print(f"Error converting image to JPEG: {e}")
        return False


# Function to process images in the input folder
def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Counter for processed images
    num_processed_images = 0

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".heic")):
            num_processed_images += 1
            input_path = os.path.join(input_folder, filename)

            try:
                # Check file size before conversion
                if os.path.getsize(input_path) == 0:
                    print(f"Skipping processing of '{filename}': File size is 0.")
                    continue

                # Convert HEIC to JPEG
                if filename.lower().endswith(".heic"):
                    # Create a temporary directory for conversion
                    temp_dir = tempfile.mkdtemp()
                    if not os.path.exists(temp_dir):
                        print(
                            f"Error creating temporary directory for HEIC conversion."
                        )
                        continue  # Skip processing if temp directory creation fails
                    temp_jpg_path = os.path.join(
                        temp_dir, f"{os.path.splitext(filename)[0]}.jpg"
                    )
                    success = convert_heic_to_jpg(input_path, temp_jpg_path)
                    if not success:
                        print(
                            f"Skipping processing of '{filename}' due to conversion error."
                        )
                        shutil.rmtree(temp_dir)
                        continue  # Skip processing if conversion failed
                    input_path = temp_jpg_path

                # Convert image to JPEG
                temp_jpg_path = os.path.join(
                    tempfile.gettempdir(), f"{os.path.splitext(filename)[0]}.jpg"
                )
                success = convert_to_jpg(input_path, temp_jpg_path)
                if not success:
                    print(
                        f"Skipping processing of '{filename}' due to conversion error."
                    )
                    continue  # Skip processing if conversion failed

                # Read the JPEG image
                image = cv2.imread(temp_jpg_path)
                if image is None:
                    print(f"Error reading JPEG image '{filename}'")
                    continue  # Skip processing if image read failed

                # Detect faces in the image
                faces = detect_faces(image)
                if len(faces) == 0:
                    print(f"No faces detected in image '{filename}'")
                    continue  # Skip processing if no faces detected

                # Process each detected face
                for i, (x, y, w, h) in enumerate(faces):
                    # Crop the face to 512x512
                    face_crop = image[y : y + h, x : x + w]
                    face_crop = cv2.resize(face_crop, (512, 512))

                    # Save the cropped face image to the output folder
                    output_face_path = os.path.join(
                        output_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg"
                    )
                    cv2.imwrite(output_face_path, face_crop)

            except Exception as e:
                print(f"Error processing image '{filename}': {e}")

    print(f"Processed {num_processed_images} images.")


# Define input and output folders
input_folder = "input"
output_folder = "output"

# Process images
process_images(input_folder, output_folder)

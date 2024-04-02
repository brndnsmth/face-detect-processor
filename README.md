# Face Detection for DreamBooth

This Python script is designed to process images, detect faces, and crop them for use with DreamBooth.

## Usage

1. Place the images you want to process in the `input` folder.
2. Run the script `main.py`.
3. Processed images with detected faces will be saved in the `output` folder.

The script supports the following file types for image processing:

- .jpg
- .jpeg
- .png
- .bmp
- .heic

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Pillow
- pyheif

## Setup Steps


1. Clone git repo.

```
git clone https://github.com/brndnsmth/face-detect-processor.git
cd face-detect-processor
```

2. **Create a Virtual Environment:** Utilize Python 3 to create a virtual environment for this project. This step ensures a clean and isolated environment for installing dependencies.

```
python3 -m venv .venv
```

3. **Activate the Virtual Environment:** Activate the virtual environment to isolate the project dependencies from other Python installations on your system.

```
source .venv/bin/activate
```

4. **Install Dependencies:** While inside the virtual environment, install the required dependencies specified in the requirements.txt file using pip.

```
pip install -r requirements.txt
```

5. **Add Images:** Create `input` folder and add images to directory.

```
touch input
```

6. **Run the Script:** Execute the following command to run the script:

```
python main.py
```

7. **Check Output:** Check processed images. Adjust the following in `main.py`, as needed:

```
scaleFactor=1.05, minNeighbors=6, minSize=(100, 100)
```

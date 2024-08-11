# Live Gender Detector

This program uses **AI** and **machine learning** techniques along with the laptop camera 
to detect, identify, and **track faces** in real-time, while also determining the 
**gender** of the detected faces.

## Table of contents

- [Best features](README.md/#best-features)
- [Installation steps](README.md/#how-to-run-the-project)
  - [Prerequisites](README.md/#prerequisites)
  - [Running the project](README.md/#running-the-project)
- [The process](README.md/#the-process)
  - [Built with](README.md/#built-with)
  - [Structure](README.md/#structure)
- [Author](README.md/#author)

## Best features

- Real-time face detection and tracking
- AI-based face detection
- Machine learning-based gender detection

## Installation steps
Follow these steps to set up and run the project on your local machine.

### Prerequisites
Ensure Python 3.12 is installed on your system. You can download it from the 
[official Python website](https://www.python.org/downloads/) or via the 
Microsoft Store. 
Then, install the required libraries from your terminal using this command:

    pip install opencv-python deepface numpy pandas sklearn


### Running the project
After installing the libraries, clone the project from GitHub and run `Main.py`. 
The terminal will display the model's accuracy, and the camera feed will show 
detected faces along with gender predictions. The terminal also logs newly 
detected and removed faces.

## The process

### Built with

- Python 3.12
- **IDE:** PyCharm Professional 2024.1.4
- **OpenCV:** For AI-based face detection
- **DeepFace:** For gender detection using machine learning
- **Sklearn:** For training machine learning models
- **Pandas:** For data preparation
- **Numpy:** For handling data embeddings

### Structure
The project includes 4 files and 1 directory:
- `Main.py`: Initializes and runs the program.
- `Face.py`: Manages face detection and tracking.
- `SelectBestModel.ipynb`: Trains and evaluates gender detection models.
- `haarcascade_frontalface_default.xml`: OpenCV’s pre-trained classifiers for face detection.
- `data` directory: Contains a CSV file with image embeddings for model training.

The code is well-commented to explain the process.

## Author
Crafted with passion by Kaarlo Caballero.






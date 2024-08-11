# Live Gender Detector

This program utilizes **AI**, **machine learning** techniques and the laptop camera 
to detect, identify, and **track faces** within the frame, while also determining the 
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
- [License]

## Best features

The program most important features are:
- Following and identifying faces
- Detecting faces using AI
- Detecting face's gender using Machine Learning

## Installation steps
These instructions will get you a copy of the project up and running on your local 
machine for development and testing purposes.

### Prerequisites
First you need to install and set up python 3.12, so you can go to this 
[link](https://www.python.org/downloads/) or download directly from Microsoft Store.
After that these are the libraries needed to install to run the project in your 
Windows CMD today, 08 of August 2024:
    
    pip install opencv-python
    pip install deepface
    pip install numpy
    pip install pandas
    pip install sklearn


### Running the project
After installing these libraries you can clone this project from GitHub and run 
the  project from  the Main.py file. When the program starts you will see in the 
terminal the accuracy of the Machine learning model trained by the program, in 
another windows there is going to be the camera frame, and in the terminal you will 
see the new faces detected and the deleted ones. 

## The process

### Built with

- Python 3.12
- IDE: Pycharm professional version 2024.1.4
- OpenCV: to detect faces using **AI**
- DeepFace: to detect the faces gender using **machine learning models**
- Sklearn: to train the **machine learning models**
- Pandas: to clean and prepare the **data** for the **machine learning** model training
- Numpy: to manipulate the embeddings for the input in the **machine learning model**

### Structure
The project is composed of 4 files(Face.py, haarcascade_frontalface_default.xml, 
Main.py and SelectBestModel.ipynb) and 1 directory(data)
- **Face.py:** this python file contains the face class and the methods used to detect and manage the faces in the frame
- **Main.py:** this python file is the one that starts the program and prepares everything for the execution of the methods in Face.py
- **SelectBestModel.ipynb:** this Jupyter notebook file contains the training and analysis of the different machine learning models considered for the gender detection task, this is where we test the accuracy of each model 
- **haarcascade_frontalface_default.xml:** this xml file contains teh weak classifiers already provided by OpenCV to detect faces in the frame
- **data directory:** this directory contains only one file which is a csv file that contains different images of males and females and its embeddings processed using DeepFace's transform function

Face.py, Main.py and SelectBestModel.ipynb have comments that explain exactly the process followed during the program.

## Author
Made with passion and effort by Kaarlo Caballero

## License






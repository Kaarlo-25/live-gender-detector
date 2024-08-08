import random
import cv2
import numpy as np
from deepface import DeepFace

face_detector = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')  # initialize the haar cascade face detector


class Face:  # create a class to manage all faces detected
    detected_faces = []  # a list that will contain all faces detected in the current frame
    face_ID = 0  # an ID that will identify every face

    def __init__(self, center: tuple, frame_counter, gender: str, color=None):
        Face.face_ID += 1   # add one to the faceID, so each new face has a different ID
        self.face_ID = Face.face_ID # faceID for the new detected face
        self.center = center    # center of the rectangle of the new detected face
        self.last_frame = frame_counter # last frame in which the detected face has appeared
        self.gender = gender    # gender of the new detected face
        self.color = color  # color of the rectangle around the detected face

    def __str__(self):  # informal representation of a face
        return f'[{self.face_ID}, {self.center}, {self.gender}, {self.color}]'


def delete_face(frame_counter):  # if a face in detected faces is not detected after 30 frames(around a second), it is deleted from the detected faces
    for face in Face.detected_faces:
        if frame_counter - face.last_frame > 30:
            Face.detected_faces.remove(face)
            print(f"deleted face: {face}")


def predict_gender(frame, x, y, width, height, model):
    frame = frame[y:y + height, x:x + width]    # crop the frame, so it only has they face already detected
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # change the image from RGB format to BGR format so DeepFace can process it
    frame = np.array(frame)  # change the image to numpy array format so DeepFace can process it
    embedding_objs = DeepFace.represent(img_path=frame,
                                        model_name='Facenet512',
                                        detector_backend='opencv',
                                        enforce_detection=False
                                        )   # use DeepFace to change the image from numpy array to a characteristics vector that can be processed by our model
    embedding = embedding_objs[0]["embedding"]  # get the characteristics vector
    return model.predict(np.array(embedding).reshape(1, -1))[0]   # use our model to predict the gender of the face detected


def detect_faces(frame, frame_counter, model):
    faces = face_detector.detectMultiScale(frame)  # detect faces in the frame
    for (x, y, width, height) in faces: # for every face get, x, y, width and height of teh rectangle around it
        if 150 <= width <= 370 or 150 <= height <= 370:  # threshold of detected face rectangle size
            center = (int(x + width / 2), int(y + height / 2))  # Center of the rectangle
            face = identify_faces(center, frame_counter, frame, x, y, width, height, model)  # verify if this face has already been detected
            if face is None:    # if this face is new
                face = new_face(center, frame_counter, frame, x, y, width, height, model)   # create a new face
            delete_face(frame_counter)  # delete faces that are not in the frame for 30 frames(around a second)
            cv2.rectangle(frame, (x, y), (x + width, y + height), face.color, 2)  # draw rectangle in frame
            cv2.putText(frame, f"{face.face_ID}-{face.gender}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, face.color, 2)  # draw face_ID and gender on the rectangle
        else:
            continue
    return frame  # return de frame fully processed


def min_distance_face(center, frame_counter):
    face_distances = [] # list that will save the index fo the face in the list and the distance to the current detected face
    for idx, face in enumerate(Face.detected_faces):    # for every face in detected faces, get index in the list and face
        face_distances.append([idx, abs(face.center[0] - center[0]) + abs(face.center[1] - center[1])])  # add to faces_distances a list that has
                                                                                                         # the index of the face in detected faces
                                                                                                         # and the manhattan distances
    closest_face = min(face_distances, key=lambda element: element[1])  # get the closest face to the current detected face
    if closest_face[1] >= 60:   # if the distance is more than 60
        return None
    else:
        update_face(Face.detected_faces[closest_face[0]], center, frame_counter)    # update the values needed in the current detected face
        return Face.detected_faces[closest_face[0]] # return the face current detected face


def new_face(center, frame_counter, frame, x, y, width, height, model):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))    # select a random color for the rectangle
    gender = predict_gender(frame, x, y, width, height, model)  # predict the gender of the face with predict_gender()
    face = Face(center, frame_counter, gender, color)   # create a new face
    print(f"new face: {face}")  # print new face to know what is happening
    Face.detected_faces.append(face)    # add new face to detected faces
    return face


def update_face(face, center, frame_counter):
    face.center = center    # update the center value of the current detected face using the new center
    face.last_frame = frame_counter # update the last frame of the current detected face using the current frame
    return face


def identify_faces(center, frame_counter, frame, x, y, width, height, model):
    if not Face.detected_faces:  # if detected faces is empty, create and return a new face
        return new_face(center, frame_counter, frame, x, y, width, height, model)
    else:
        return min_distance_face(center, frame_counter)  # check the closest face

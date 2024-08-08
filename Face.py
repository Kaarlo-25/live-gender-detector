import random
import cv2
import numpy as np
from deepface import DeepFace


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # initialize the classifier


class Face:
    detected_faces = []
    face_ID = 0

    def __init__(self, center: tuple, frame_counter, gender: str, color=None):
        Face.face_ID += 1
        self.face_ID = Face.face_ID
        self.center = center
        self.last_frame = frame_counter
        self.gender = gender
        self.color = color

    def __str__(self):
        return f'[{self.face_ID}, {self.center}, {self.gender}, {self.color}]'


def delete_face(frame_counter):
    for face in Face.detected_faces:
        if frame_counter - face.last_frame > 30:
            Face.detected_faces.remove(face)
            print(f"deleted face: {face}")


def predict_gender(frame, x, y, width, height, model):
    frame = frame[y:y + height, x:x + width]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = np.array(frame)
    embedding_objs = DeepFace.represent(img_path=frame,
                                        model_name='Facenet512',
                                        detector_backend='opencv',
                                        enforce_detection=False
                                        )
    embedding = embedding_objs[0]["embedding"]
    gender = model.predict(np.array(embedding).reshape(1, -1))[0]
    return gender


def detect_faces(frame, frame_counter, model):
    faces = face_detector.detectMultiScale(frame)  # detect faces in frame
    for (x, y, width, height) in faces:
        if 150 <= width <= 370 or 150 <= height <= 370:  # threshold of face size
            center = (int(x + width / 2), int(y + height / 2))  # Center of the face
            face = identify_faces(center, frame_counter, frame, x, y, width, height, model)
            if face is None:
                face = new_face(center, frame_counter, frame, x, y, width, height, model)
            #gender = predict_gender(frame, x, y, width, height)
            delete_face(frame_counter)  # delete faces that are not in the camara for 30 frames
            cv2.rectangle(frame, (x, y), (x + width, y + height), face.color, 2)  # draw rectangle in frame
            cv2.putText(frame, f"{face.face_ID}-{face.gender}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, face.color, 2)  # draw face_ID in frame
            #cv2.putText(frame, f"{face.gender}", (x, y - 10),
            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, face.color, 2)  # draw face_ID in frame
        else:
            continue
    return frame


def min_distance_face(center, frame_counter):
    face_distances = []
    for idx, face in enumerate(Face.detected_faces):
        face_distances.append([idx, abs(face.center[0] - center[0]) + abs(face.center[1] - center[1])])
    closest_face = min(face_distances, key=lambda element: element[1])
    if closest_face[1] >= 60:
        return None
    else:
        update_face(Face.detected_faces[closest_face[0]], center, frame_counter)
        return Face.detected_faces[closest_face[0]]


def new_face(center, frame_counter, frame, x, y, width, height, model):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    gender = predict_gender(frame, x, y, width, height, model)
    face = Face(center, frame_counter, gender, color)
    print(f"new face: {face}")
    Face.detected_faces.append(face)
    return face


def update_face(face, center, frame_counter):
    face.center = center
    face.last_frame = frame_counter
    return face


def identify_faces(center, frame_counter, frame, x, y, width, height, model):
    if not Face.detected_faces:
        return new_face(center, frame_counter, frame, x, y, width, height, model)
    else:
        return min_distance_face(center, frame_counter)

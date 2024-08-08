import csv
import cv2
import numpy as np
from deepface import DeepFace

new_data = []
i = 0
while i < 1134:
    directory = f"C:\\Users\\valko\\OneDrive\\Documentos\\data\\woman\\face_{i}.jpg"
    image = cv2.imread(directory)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    embedding_objs = DeepFace.represent(img_path=image,
                                        model_name='Facenet512',
                                        detector_backend='opencv',
                                        enforce_detection=False
                                        )
    embedding = embedding_objs[0]["embedding"]
    data2save = [f"face_{i}", "female"] + embedding
    new_data.append(data2save)
    print(data2save)
    i += 1

filename = "C:\\Users\\valko\\OneDrive\\Documentos\\Images_data.csv"
with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new_data)
    print("saved")

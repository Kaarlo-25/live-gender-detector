import cv2
from Face import detect_faces
from GenderModelPreparation import train_gender_model

camara = cv2.VideoCapture(0)
if not camara.isOpened():  # checks if camara is working
    print("Camara not working.")
    exit()
frame_counter = 0
model = train_gender_model()
while True:
    ret, frame = camara.read()  # returns the frame
    if not ret:  # if there's no frame
        print("No frame received.")
        break

    cv2.imshow('camara', detect_faces(frame, frame_counter, model))
    frame_counter += 1

    if cv2.waitKey(1) == ord('q'):
        camara.release()
        cv2.destroyAllWindows()
        break

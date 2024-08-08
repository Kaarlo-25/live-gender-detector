import cv2
from Face import detect_faces
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_gender_model():
    data = pd.read_csv('data\\Images_data.csv') # create dataframe from the data gathered
    y = data.iloc[:, 1].values  # establish y values, which will be the results to predict
    x = data.iloc[:, 2:].values # establish x values, which will be the entering values from which the model will predict y
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=0, shuffle=True)  # divide data in train and test

    clf = SVC(C=3, kernel='rbf')    # create the instance of the model with the hyperparameters that maximize its performance
    clf.fit(train_x, train_y)   # train the model
    y_pred = clf.predict(test_x)    # use the x values from the test data set to evaluate its performance after training
    accuracy = round(accuracy_score(test_y, y_pred) * 100, 2)   # calculate the accuracy, based on how many predictions were correct
    print(f"Accuracy: {accuracy}%\n")   # print the accuracy result
    return clf  # return de trained model


camara = cv2.VideoCapture(0)    # start the camera of the computer
if not camara.isOpened():  # checks if camara is working
    print("Camara not working.")    # print a message to know what is happening
    exit()  # close camera
frame_counter = 0   # variable to know in what frame are we currently, so we can know when a person is not on the frame
model = train_gender_model()    # we train and get de model ready to use
while True:
    ret, frame = camara.read()  # returns the camera frame and a boolean indicating if its working
    if not ret:  # if there's no frame
        print("No frame received.") # print a message to know what is happening
        break   # exit the loop

    cv2.imshow('camara', detect_faces(frame, frame_counter, model)) # show image once it's processed by detect_faces()
    frame_counter += 1

    if cv2.waitKey(1) == ord('q'):  # if 'q' button is pressed, close the camera, and close all windows opened by cv2
        camara.release()
        cv2.destroyAllWindows()
        break

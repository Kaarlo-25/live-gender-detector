import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_gender_model():
    data = pd.read_csv('C:\\Users\\valko\\PycharmProjects\\live-gender-detector\\data\\Images_data.csv')
    y = data.iloc[:, 1].values
    x = data.iloc[:, 2:].values
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=0, shuffle=True)

    clf = SVC(C=3, kernel='rbf')
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    accuracy = round(accuracy_score(test_y, y_pred) * 100, 2)
    print(f"Accuracy: {accuracy}%\n")
    return clf


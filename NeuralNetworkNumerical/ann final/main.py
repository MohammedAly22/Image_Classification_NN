import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('data.csv')


def dataSetAnalysis(df):
    # view starting values of data set
    print("Dataset Head")
    print(df.head(3))
    print("=" * 30)

    # View features in data set
    print("Dataset Features")
    print(df.columns.values)
    print("=" * 30)

    # View How many samples and how many missing values for each feature
    print("Dataset Features Details")
    print(df.info())
    print("=" * 30)

    # view distribution of numerical features across the data set
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 30)

    # view distribution of categorical features across the data set
    print("Dataset Categorical Features")
    print(df.describe(include=['O']))
    print("=" * 30)


dataSetAnalysis(dataset)

X = dataset.iloc[:, 2:32]  # [all rows, col from index 2 to the last one excluding 'Unnamed: 32']
y = dataset.iloc[:, 1]  # [all rows, col one only which contains the classes of cancer]

print("Before encoding: ")
print(y[100:110])

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

print("\nAfter encoding: ")
print(y[100:110])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()  # Initialising the ANN

classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=1, epochs=100)

classifier.save('breast_cancer_model.h5')  # Save trained ANN

# classifier = load_model('breast_cancer_model.h5')  #Load trained ANN

y_pred = classifier.predict(X_test)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")


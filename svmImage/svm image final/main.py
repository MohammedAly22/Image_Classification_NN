import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
import timeit

start = timeit.default_timer()
dir = 'C://Users//as//PycharmProjects//FinalNN//natural_images'
categories = ['cat', 'dog', 'airplane', 'car', 'person', 'flower']
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath, 0)
        try:
            pet_img = cv2.resize(pet_img, (50, 50))
            image = np.array(pet_img).flatten()
            data.append([image, label])
        except Exception as e:
            pass

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.01)
model = svm.SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)
categories = ['cat', 'dog', 'airplane', 'car', 'person']
print('accuracy : ', int(accuracy * 100), '%')
print('prediction is : ', categories[prediction[0]])
mypet = xtest[0].reshape(50, 50)
plt.imshow(mypet, cmap='gray')
plt.show()

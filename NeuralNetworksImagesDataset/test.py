############ IMPORT PACKAGES ###############
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix

############# LOADING OUR MODEL ###############
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
model = load_model('final-model.model')

############# READING AND PREPARING IMAGE ###############
def prepareImage(imagePath):
    image_size = 150
    img_array = cv.imread(imagePath)
    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
    new_array = cv.resize(img_array, (image_size, image_size))
    plt.imshow(new_array)
    return new_array.reshape(-1, image_size, image_size, 3)

############# TESTING MODEL ###############
prediction = model.predict(prepareImage("F:/plane.jpg"))
index = prediction.argmax()
print(f"prediction is: {class_names[index]}")
plt.show()

############### IMPORT IMPORTANT PACKAGES ####################
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import cv2 as cv
import pandas as pd

############### IMAGE PATHS AND CLASS NAMES ##################
train_dir = "F:/programming/python/data sets/dataset/training"
validation_dir = "F:/programming/python/data sets/dataset/validation"
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

train_datgen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
val_datgen = ImageDataGenerator(rescale=1./255)

train_generator = train_datgen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)
val_generator = val_datgen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)
# prepare image function
def prepareImage(imagePath):
    image_size = 150
    img_array = cv.imread(imagePath)
    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
    new_array = cv.resize(img_array, (image_size, image_size))
    plt.imshow(new_array)
    plt.show()
    return new_array.reshape(-1, image_size, image_size, 3)

############### BUILDING OUR MODEL ##################
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))

############### COMPILING AND FITTING OUR MODEL ##################
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=276,
    epochs=30,
    validation_data=val_generator,
    validation_steps=66
)

############### SAVE MODEL ##################
model.save("final-model.model")

############# EVALUATING AND PLOTTING  ###############
print(model.evaluate(val_generator))
# loss plotting
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()
# accuracy plotting
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

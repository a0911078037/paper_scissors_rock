import joblib
import keras.optimizers
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict

directory = './test_data'
labels = ['paper', 'scissors', 'rock']
nb = len(labels)

def input_target_split(train_dir, labels):
    dataset = []
    count = 0
    for label in labels:
        folder = os.path.join(train_dir, label)
        for image in os.listdir(folder):
            img = load_img(os.path.join(folder, image), target_size=(150, 150))
            img = img_to_array(img)
            img = img / 255.0
            img = img.flatten()
            dataset.append((img, count))
        count += 1
    random.shuffle(dataset)
    X, y = zip(*dataset)

    return np.array(X), np.array(y)

X, y = input_target_split(directory, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'backup/DecisionTree.joblib')



result = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(result)
print(result.mean())

# datagen = ImageDataGenerator(horizontal_flip=True,
#                              vertical_flip=True,
#                              rotation_range=20,
#                              zoom_range=0.2,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
#                              shear_range=0.1,
#                              fill_mode="nearest")
#
# testgen = ImageDataGenerator()
#
# datagen.fit(X_train)
# testgen.fit(X_test)
#
# y_train = np.eye(nb)[y_train]
# y_test = np.eye(nb)[y_test]
#
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
# model.add(MaxPooling2D(2, 2))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(2, 2))
# model.add(Flatten())
# model.add(Dense(units=512, activation='relu'))
# model.add(Dense(units=3, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# filepath = "model_cnn_final.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',
#                              save_weights_only=False)
#
# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)
#
# # learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
# #                                             patience=3,
# #                                             verbose=1,
# #                                             factor=0.2,
# #                                             min_lr=0.00001)
#
# callbacks_list = [
#     checkpoint,
#     early_stopping,
#     #         learning_rate_reduction
# ]
#
# hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
#                            validation_data=testgen.flow(X_test, y_test, batch_size=32),
#                            epochs=50,
#                            callbacks=callbacks_list,
#                            steps_per_epoch=1000,
#                            validation_steps=10)

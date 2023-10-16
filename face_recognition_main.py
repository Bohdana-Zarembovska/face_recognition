import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

image_folder_1 = '1_face_recognition_dataset\\Extracted_Faces\\Extracted_Faces'
image_folder_2 = '1_face_recognition_dataset\\Face_Data\\Face_Dataset'

folders_1 = os.listdir(image_folder_1)
folders_2 = os.listdir(image_folder_2)

common_folders = set(folders_1) & set(folders_2)

num_classes = len(common_folders)

images = []
labels = []

for n, folder in enumerate(common_folders):
    folder_path_1 = os.path.join(image_folder_1, folder)
    folder_path_2 = os.path.join(image_folder_2, folder)

    files_1 = os.listdir(folder_path_1)
    files_2 = os.listdir(folder_path_2)

    for file_1, file_2 in zip(files_1, files_2):
        image_path_1 = os.path.join(folder_path_1, file_1)
        image_path_2 = os.path.join(folder_path_2, file_2)

        img_1 = cv2.imread(image_path_1)
        img_2 = cv2.imread(image_path_2)

        img_1 = cv2.resize(img_1, (64, 64))
        img_2 = cv2.resize(img_2, (64, 64))

        images.append(img_1)
        labels.append(n)  

images = np.array(images)
labels = np.array(labels)

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def create_siamese_network(input_shape):
    input = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    encoded = Dense(128)(x)

    model = Model(input, encoded)
    return model

input_shape = (64, 64, 3)

input_left = Input(shape=input_shape)
input_right = Input(shape=input_shape)

siamese_model = create_siamese_network(input_shape)
encoded_left = siamese_model(input_left)
encoded_right = siamese_model(input_right)

distance = Lambda(euclidean_distance)([encoded_left, encoded_right])

model = Model(inputs=[input_left, input_right], outputs=distance)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit([images, images], labels, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stopping])

for epoch, (train_acc, train_loss, val_acc, val_loss) in enumerate(zip(
    history.history['accuracy'],
    history.history['loss'],
    history.history['val_accuracy'],
    history.history['val_loss']
), start=1):
    print(f'Epoch {epoch}:')
    print(f'Training Accuracy: {train_acc:.4f}, Training Loss: {train_loss:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}')

y_pred = model.predict([images, images])
y_pred = (y_pred < 0.5).astype(int)
accuracy = np.mean(np.equal(labels, y_pred))
print(f'Accuracy: {accuracy * 100:.2f}%')
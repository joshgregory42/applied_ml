import numpy as np
import pandas as pd
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

# Doing experiment tracking in Weights and Biases, which is the same that I'm using for my thesis. Trying to use this to get some experience using it
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Also going to hyperparameter optimize using Optuna, again this is the same library that I'm using as in my thesis, so I'm going to use it here to get familiar with it.
import optuna

wandb.login()
wandb.init()


tf.config.list_physical_devices('GPU')


def pre_process_images(data_directory):
    features = []
    labels = []

    for color in ['red', 'yellow', 'green']:
        color_path = os.path.join(data_directory, color)

        # Get all files in color directory

        for file in os.listdir(color_path):
            img_path = os.path.join(color_path, file)
                
            # Read in image
            img = np.array(Image.open(img_path).convert('RGB'))

            image_crop = np.copy(img)
            row_crop = 7
            col_crop = 8
            image_crop = img[row_crop:-row_crop, col_crop:-col_crop, :]

            img_resized = cv2.resize(image_crop, (32, 32))

            # Not flattening image here, since the CNN will take in 32x32x3 images
            img_standardized = tf.image.per_image_standardization(img_resized)

            features.append(img_standardized)

            
            # Append the label as well
            labels.append(color)

    features = np.array(features)

    # Convert strings of colors to integer values
    light_dict = {'red': 0, 'yellow': 1, 'green': 2}
    labels = np.array([light_dict[label] for label in labels])
    features, labels = shuffle(features, labels, random_state=42)

    return features, labels


def create_base_cnn(input_shape=(32, 32, 3), num_classes=3):
    model = models.Sequential([

        # First convolutional layer
        layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last', name='conv_1', activation='relu'
        ),
        
        # Second convolutional layer
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', data_format='channels_last', name='conv2'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', data_format='channels_last', name='conv3'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Fully Connected Layers
        layers.Flatten(name='Flatten'),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
    

def train_model(X_train, y_train, X_test, y_test):

    # Data augmentation to prevent overfitting
    data_augmentation = models.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomFlip('vertical'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Create base model without data autmentation
    model = create_base_cnn()

    model = models.Sequential([
        data_augmentation,
        model
    ])

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        restore_best_weights=True,
        patience=5
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,
        patience=5, 
        min_lr=0.001
    )

    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) # same as `tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')`

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, WandbMetricsLogger()]
    )

    return model


def main():
    
    # Example of generating random input data for testing
    import numpy as np
    
    X, y = pre_process_images('traffic_light_images_shah/data/') 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

if __name__ == "__main__":
    main()
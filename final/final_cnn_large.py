import numpy as np
import pandas as pd
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import seaborn as sns

# Doing experiment tracking in Weights and Biases, which is the same that I'm using for my thesis. Trying to use this to get some experience using it
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Also going to hyperparameter optimize using Ray Tune, again this is the same library that I'm using as in my thesis, so I'm going to use it here to get familiar with it.
import ray
from ray import tune
from ray.tune import Trainable

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
            img_resized = cv2.resize(image_crop, (32, 32))

            # Not flattening image here, since the CNN will take in 32x32x3 images
            img_standardized = tf.image.per_image_standardization(img_resized)

            features.append(img_standardized)
            
            # Append the label
            labels.append(color)

    features = np.array(features)

    # Convert strings of colors to integer values
    light_dict = {'red': 0, 'yellow': 1, 'green': 2}
    labels = np.array([light_dict[label] for label in labels])
    features, labels = shuffle(features, labels, random_state=42)

    return features, labels


def create_cnn(input_shape=(32, 32, 3), num_classes=3):
    model = models.Sequential([

        # Input layer
        layers.InputLayer(shape=input_shape, name='input_layer'),

        # Data augmentation steps to prevent overfitting
        layers.RandomFlip('horizontal', name='horiz_flip'),
        layers.RandomFlip('vertical', name='vert_flip'),
        layers.RandomRotation(0.1, name='random_rotation'),
        layers.RandomZoom(0.1, name='random_zoom'),

        # CNN implementation
        # First convolutional layer
        layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last', name='conv_1', activation='relu'
        ),
        
        # Second convolutional layer
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', data_format='channels_last', name='conv_2'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.MaxPooling2D((2, 2), name='max_pool_1'),
        layers.Activation('relu', name='relu_1'),
        
        # Third convolutional layer
        layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', data_format='channels_last', name='conv_3'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.MaxPooling2D((2, 2), name='max_pool_2'),
        layers.Activation('relu', name='relu_2'),
        
        # Flatten and Fully Connected Layers
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.5, name='dropout_1'),
        layers.Dense(num_classes, activation='softmax', name='output_layer')

    ])

    return model
    

def train_model(X_train, y_train, X_test, y_test):

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        restore_best_weights=True,
        patience=20
    )

    model = create_cnn()
    
    model.compile(optimizer=tf.keras.optimizers.AdamW(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) # same as `tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')`

    model.summary()

    # plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True, dpi=1000)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1000,
        batch_size=512,
        callbacks=[early_stopping, WandbMetricsLogger()]
    )

    return model


def main():
    
    X, y = pre_process_images('traffic_lights_large/data/') 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_test shape: {y_test.shape}")

    model = train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # Save model in .h5 format
    model.save('models/cnn_large.keras')

    # Predict on the test set
    y_pred = np.argmax(model.predict(X_test), axis=1)

    confmat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confmat, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(['red', 'yellow', 'green']), yticklabels=np.unique(['red', 'yellow', 'green']))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('CNN Confusion Matrix (Large Dataset)')
    plt.savefig('images/cnn_conf_mat_large.png', dpi=1000)

    labels = ['red', 'yellow', 'green']
    print(classification_report(y_pred=y_pred, y_true=y_test))
    # plt.show()

if __name__ == "__main__":
    main()
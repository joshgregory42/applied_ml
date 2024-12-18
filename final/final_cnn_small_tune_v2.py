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
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

wandb.login()
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
            
            # Append the label
            labels.append(color)

    features = np.array(features)

    # Convert strings of colors to integer values
    light_dict = {'red': 0, 'yellow': 1, 'green': 2}
    labels = np.array([light_dict[label] for label in labels])
    features, labels = shuffle(features, labels, random_state=42)

    return features, labels

def train_model(config, input_shape=(32, 32, 3), num_classes=3):
    
    X, y = pre_process_images('/home/josh/applied_ml/final/traffic_light_images_shah/data') 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make sure the config has the right keys
    filters = config['filters'].value if isinstance(config['filters'], tune.search.sample.Categorical) else config['filters']
        
    model = models.Sequential([
        
        # Input layer
        layers.InputLayer(input_shape=input_shape, name='input_layer'),

        # Data augmentation steps to prevent overfitting
        layers.RandomFlip('horizontal', name='horiz_flip'),
        layers.RandomFlip('vertical', name='vert_flip'),
        layers.RandomRotation(0.1, name='random_rotation'),
        layers.RandomZoom(0.1, name='random_zoom'),

        # CNN implementation
        # First convolutional layer
        layers.Conv2D(filters=config['filters'], kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last', name='conv_1', activation='relu'
        ),
        
        # Second convolutional layer
        layers.Conv2D(filters=config['filters']*2, kernel_size=(3, 3), padding='same', data_format='channels_last', name='conv_2'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.MaxPooling2D((2, 2), name='max_pool_1'),
        layers.Activation('relu', name='relu_1'),
        
        # Third convolutional layer
        layers.Conv2D(filters=config['filters']*4, kernel_size=(3, 3), padding='same', data_format='channels_last', name='conv_3'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.MaxPooling2D((2, 2), name='max_pool_2'),
        layers.Activation('relu', name='relu_2'),
        
        # Flatten and Fully Connected Layers
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.5, name='dropout_1'),
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=config['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']) # same as `tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')`
    
    # Initialize W&B
    wandb.init(project='cnn_large_tune', config=config, reinit=True)
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        restore_best_weights=True,
        patience=20
    )
    
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1000,
        batch_size=config['batch_size'],
        callbacks=[early_stopping, WandbMetricsLogger()]
    )
    
    best_val_accuracy = max(history.history['val_accuracy'])
    
    ray.train.report({'accuracy': best_val_accuracy})
    
 
def main():
    
    ray.init()
    wandb.login()
    
    metric = 'accuracy'
    mode = 'max'
    
    search_space = {
        'learning_rate': tune.loguniform(1e-5, 1e-2),
        'filters': tune.choice([16, 32, 64]),
        'batch_size': tune.choice([64, 128, 256, 512])
    }
    
    # Configure scheduler and search algorithm
    scheduler = ASHAScheduler(
        max_t=1000,  # max epochs
        grace_period=10,
        reduction_factor=2,
        # metric=metric,
        # mode=mode
    )
    search_alg = OptunaSearch(metric=metric, mode=mode)

    # Start Ray Tune search
    analysis = tune.run(
        train_model,
        config=search_space,
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=20,
        metric='accuracy',
        mode='max',
        resources_per_trial={'cpu':18, 'gpu':1}
    )

    # Print best hyperparameters and results
    print(f'Best hyperparameters found: {analysis.best_config}')
    print(f'Best val. accuracy: {analysis.best_result['accuracy']}')
    
    # Load data for final model
    X, y = pre_process_images('/home/josh/applied_ml/final/traffic_light_images_shah/data')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_config = analysis.best_config
    
       # Ensure best_config uses integer values
    best_config['filters'] = int(best_config['filters'])

    # Create and train final model
    final_model = models.Sequential([
        # (Same model architecture as in train_model)
        layers.InputLayer(input_shape=(32, 32, 3), name='input_layer'),
        layers.RandomFlip('horizontal', name='horiz_flip'),
        layers.RandomFlip('vertical', name='vert_flip'),
        layers.RandomRotation(0.1, name='random_rotation'),
        layers.RandomZoom(0.1, name='random_zoom'),
        layers.Conv2D(filters=best_config['filters'], kernel_size=(5, 5), 
                     strides=(1, 1), padding='same', 
                     data_format='channels_last', name='conv_1', activation='relu'),
        layers.Conv2D(filters=best_config['filters']*2, kernel_size=(3, 3), 
                     padding='same', data_format='channels_last', name='conv_2'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.MaxPooling2D((2, 2), name='max_pool_1'),
        layers.Activation('relu', name='relu_1'),
        layers.Conv2D(filters=best_config['filters']*4, kernel_size=(3, 3), 
                     padding='same', data_format='channels_last', name='conv_3'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.MaxPooling2D((2, 2), name='max_pool_2'),
        layers.Activation('relu', name='relu_2'),
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.5, name='dropout_1'),
        layers.Dense(3, activation='softmax', name='output_layer')
    ])

    final_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=best_config['learning_rate']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    final_history = final_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1000,
        batch_size=best_config['batch_size']
    )

    # Save model
    final_model.save('models/cnn_large_tuned.keras')

    # Predict on the test set
    y_pred = np.argmax(final_model.predict(X_test), axis=1)

    confmat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confmat, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(['red', 'yellow', 'green']), yticklabels=np.unique(['red', 'yellow', 'green']))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('CNN Confusion Matrix (Small Dataset, Tuned)')
    plt.savefig('images/cnn_conf_mat_small_tuned.png', dpi=1000)

    labels = ['red', 'yellow', 'green']
    print(classification_report(y_pred=y_pred, y_true=y_test))
    # plt.show()

if __name__ == "__main__":
    main()
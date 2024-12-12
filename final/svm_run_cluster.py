# importing the necessary libraries
import os
import pickle
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn import tree, preprocessing
from sklearn.utils import shuffle
import seaborn as sns

sns.set()

def pre_process_images_large(data_directory):
    features = []
    labels = []

    for color in ['red', 'yellow', 'green']:
        color_path = os.path.join(data_directory, color)

        # Get all files in color directory

        for file in os.listdir(color_path):
            img_path = os.path.join(color_path, file)
                
            # Read in image
            img = cv2.imread(img_path)

            image_crop = np.copy(img)
            # row_crop = 7
            # col_crop = 8
            # image_crop = img[row_crop:-row_crop, col_crop:-col_crop, :]

            img_resized = cv2.resize(image_crop, (32, 32))

            # Resize image
            # img_resized = transform.resize(img, (32, 32))

            # Flatten image
            flat_features = img_resized.flatten()
            features.append(flat_features)

            # Append the label as well
            labels.append(color)

    features = np.array(features)

    # Convert strings of colors to integer values
    light_dict = {'red': 0, 'yellow': 1, 'green': 2}
    labels = np.array([light_dict[label] for label in labels])
    features, labels = shuffle(features, labels, random_state=42)

    return features, labels

# Prepare training data

X_large, y_large = pre_process_images_large('traffic_lights_large/data/')

X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(X_large, y_large, test_size=0.2, random_state=42)


pipe_svc = make_pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('svc', SVC())                 # Support Vector Classifier
])

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf', 'linear']}  
  
# param_grid = {'C': [0.1, 1, 10, 100, 1000],}     
grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=3, n_jobs=-1) 
  
# fitting the model for grid search 
grid_search.fit(X_train_large, y_train_large) 

# Get the best estimator from the grid search (this will be the model with optimal hyperparameters)
best_model_large = grid_search.best_estimator_

print("Optimal hyperparameters:", grid_search.best_params_)

best_model_large.fit(X_train_large, y_train_large)
accuracy = best_model_large.score(X_test_large, y_test_large)
print(f'Final accuracy on test set: {accuracy}')


## Plot learning curve with optimal hyperparameters

pipe_svc = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
                                           
train_sizes, train_scores, test_scores =\
                learning_curve(estimator=best_model_large,
                               X=X_train_large,
                               y=y_train_large,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=-1)

#print(train_sizes)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid(True)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Learning Curve")
plt.ylim([0.5, 1.03])
plt.tight_layout()
plt.savefig('learning_curve_large.png', dpi=1000)

## Create confusion matrix

y_pred_large = best_model.predict(X_test_large)
confmat_large = confusion_matrix(y_test_large, y_pred_large)


plt.figure(figsize=(8, 6))
sns.heatmap(confmat_large, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(['red', 'yellow', 'green']), yticklabels=np.unique(['red', 'yellow', 'green']))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('conf_matrix_large.png', dpi=1000)


labels = ['red', 'yellow', 'green']
print(classification_report(y_pred=y_pred_large, y_true=y_test_large))

# Save model as .pkl file

if not os.path.exists('models'):
    os.mkdir('models')

with open('models/svm_model_large.pkl', 'wb') as f:
    pickle.dump(best_model_large, f)
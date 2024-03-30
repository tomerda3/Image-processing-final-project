# Image-processing-final-project
Authors:

Tomer Damti (ID 316614098)
Email: tomerda3@ac.sce.ac.il
Phone: 0527200595
Ofri Rom (ID 208891804)
Email: ofriro@ac.sce.ac.il
Phone: 0547706109

Project Description: Gender Classification Using Local Binary Patterns (LBP) and SVM This project aims to develop a machine learning model for classifying textures in images. The model focuses on utilizing Local Binary Patterns (LBP) features and a Support Vector Machine (SVM) classifier.

Main Stages:
Data Preparation:
Loads images with various textures (man and woman free handwrite from the folders:train,val, test).
Splits data into training evaluating and testing sets for model development and evaluation.
Feature Extraction: Utilizes LBP features, capturing spatial patterns in image pixels.
Extracts LBP features from each image.
Combines LBP features with potentially useful features (like Haralick features).
Model Training:
Uses SVM, a classification algorithm.
Trains the SVM on the combined features and corresponding labels (texture classes) in the training set.
Evaluation:
Evaluates the trained model on the testing set to assess classification accuracy.
Metrics like accuracy, precision, recall, are seen throw the confusion matrix.

Model Saving (Optional):
Saves the trained SVM model for later use or deployment you can used pickle like us.

Prediction :
Uses the saved model to predict the texture class of a new image.

Explore the different LBP variants and feature extraction techniques.
use radius 1 with 8 points and radius of 3 with 24 points.

Experiment with various SVM hyperparameters for optimal performance.
This is suggested parameter grid C and gamma : 
param_grid = {'C': [0.1, 1, 10, 100],
 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

Running the Program:
install the requirment.txt file - all the packages needed is there
python classifier.py path_train path_val path_test
path_train: Path to the training set directory
path_val: Path to the validation set directory
path_test: Path to the testing set directory

Results File:
The program will create a text file named results.txt containing:
Values of the parameters that yielded the highest accuracy
Model's achieved accuracy (e.g., Accuracy: %)
Confusion Matrix for the results

Links:
confusion matrix: https://en.wikipedia.org/wiki/Confusion_matrix
Support Vector Machines (SVM):https://learnopencv.com/support-vector-machines-svm
SVM using Scikit-Learn in Python: https://learnopencv.com/svm-using-scikit-learn-in-python/
Dataset HHD_gender: https://doi.org/10.1007/978-3-030-89131-2_30
Local Binary Patterns (LBP): [https://en.wikipedia.org/wiki/Local_binary_patterns (Replace the bracketed link with the actual URL for Local Binary Patterns)](https://doi.org/10.1007/978-3-030-89131-2_30)

github link:
https://github.com/tomerda3/Image-processing-final-project

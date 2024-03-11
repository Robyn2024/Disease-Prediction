# Disease-Prediction
Train an algorithm to recognize disease symptoms and predict if a patient is at risk.

# TensorFlow Neural Network Algorithm
# Summary:
The TensorFlow Neural Network algorithm is designed to predict whether a patient is at risk of a disease based on symptoms and their severity. It utilizes a binary classification approach and employs a neural network architecture for model training and evaluation.
# Functionality:
Data Preparation:
* Merges two datasets containing disease symptoms and symptom severity.
* Converts categorical features to numerical using one-hot encoding.
* Defines the target variable indicating disease development based on symptoms.
Neural Network Model:
* The architecture consists of one hidden layer with 64 units and ReLU activation.
* Output layer has 1 unit with a sigmoid activation function.
Training:
* Trains the model using training data and plots training/validation accuracy.
Evaluation:
* Evaluate the model on the test set using metrics like confusion matrix, precision, recall, accuracy, and F1 score.
  
Usage:
* Adjust model architecture, hyperparameters, or features for improved performance.
* Monitor accuracy and F1 score to gauge model effectiveness in disease risk prediction

# DecisionTree Algorithm
# Summary:
The DecisionTree algorithm is another approach for predicting disease risk based on symptom data. It utilizes decision tree methodology to classify patients into risk categories.
# Functionality:
*Data Preparation:
  * Similar to TensorFlow algorithm, prepares data for classification.
* Decision Tree Model:
  * Constructs a decision tree based on symptom features.
  * Splits data recursively based on symptom severity to predict disease risk.
* Training:
  * Trains the decision tree model on training data.
* Evaluation:
  * Evaluates model performance using metrics like accuracy, precision, recall, and F1 score.

Usage:
* Utilize decision tree algorithm for disease risk prediction alongside or as an alternative to neural network approach.
* Experiment with different tree depths or splitting criteria to optimize model performance.

# Conclusion:
Both TensorFlow neural network and DecisionTree algorithms offer valuable tools for disease risk prediction based on symptom data. By leveraging machine learning techniques, healthcare professionals can enhance early detection and intervention strategies, leading to improved patient outcomes and resource optimization in healthcare delivery.

# Resources Used
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?resource=download

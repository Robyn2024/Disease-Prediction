# Disease-Prediction
Training an algorithm to predict whether a patient has a disease based on their symptoms is immensely beneficial in modern healthcare. This approach enables early detection of potential health issues, allowing for prompt medical intervention and personalized treatment plans. By leveraging machine learning and predictive analytics, healthcare professionals can identify patterns and risk factors, providing more accurate diagnoses and improving overall patient outcomes. The timely recognition of symptoms can significantly enhance the efficiency of healthcare systems, optimize resource allocation, and reduce the economic burden associated with prolonged or advanced-stage treatments. Additionally, predictive algorithms empower individuals by offering insights into their health risks, fostering proactive healthcare management and promoting a preventive approach to well-being.

# TensorFlow Neural Network Algorithm
# Summary:
The TensorFlow Neural Network algorithm is designed to predict whether a patient is at risk of a disease based on symptoms and their severity. It utilizes a binary classification approach and employs a neural network architecture for model training and evaluation. This algorithm was created as a backup in case the DecisionTree algorithm failed.
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
* Data Preparation:
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

# Report:
REPORT BASED ON THE THREE MODELS:

Initial Model:
F1-score: 84.58%
Accuracy: 87.09%
Hyperparameters: max_depth=10, min_samples_split=5

2nd Model:
F1-score: 98.02%
Accuracy: 98.17%
Hyperparameters: max_depth=15, min_samples_split=10

3rd Model:
F1-score: 99.58%
Accuracy: 99.59%
Hyperparameters: max_depth=20, min_samples_split=5

Observations:
The F1-scores and accuracies improve as we moved from the initial model to the 3rd model, indicating that the models are becoming more effective in capturing patterns in the data.

The 3rd model has the highest F1-score and accuracy, suggesting that it performs the best among the three.

It seems like increasing the max_depth parameter and reducing min_samples_split contribute to better performance. However, increasing model complexity (higher max_depth) may lead to overfitting, and these hyperparameters might not be optimal for all datasets.

In summary, based on the provided metrics, the 3rd model with max_depth=20 and min_samples_split=5 seems to be the best-performing model among the three.


Notes to consider: Overfitting in machine learning occurs when a model excessively captures noise and fluctuations in the training data, leading to reduced generalization to new data. This phenomenon results in a loss of predictive power, diminished ability to handle variability, increased model complexity, resource intensiveness, and limited transferability. Addressing overfitting is crucial for creating models that balance optimal training fit with the ability to generalize effectively to diverse real-world scenarios. Strategies such as regularization, cross-validation, and careful hyperparameter tuning play pivotal roles in achieving this balance.

# Conclusion:
Both TensorFlow neural network and DecisionTree algorithms offer valuable tools for disease risk prediction based on symptom data. By leveraging machine learning techniques, healthcare professionals can enhance early detection and intervention strategies, leading to improved patient outcomes and resource optimization in healthcare delivery.

# Resources Used
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?resource=download

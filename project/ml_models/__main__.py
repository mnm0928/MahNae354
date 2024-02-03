import pandas as pd
from project.ml_models.logistic_regression import LogisticRegressionModel
from project.ml_models.naive_bayes import NaiveBayesModel
from project.ml_models.svm import SVMModel
from project.preprocessing.text_cleaning import clean_text
"""
Overview:

In all models:
The cleaned text did not make much difference to ML models. In fact it lowered the accuracy.
The rows with no labels and text were removed. 
CountVectorizer or TFIDF was used to convert text to numerical data. CountVectorizer slightly gave better results.

###### Logistic Regression Model (without cleaning of text) ######
Test Accuracy: 0.8739

###### Naive Bayes Model (without cleaning of text) ######
Test Accuracy: 0.8623

###### Support Vector Machine (without cleaning of text) ######
Test Accuracy: 0.8765
"""

# read the file
df = pd.read_csv(r"/sample_data.csv")

# Apply text cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Logistic Regression Model
print("\n###### Logistic Regression Model ######")
lr_model = LogisticRegressionModel(df)
X_train, X_test, y_train, y_test = lr_model.preprocess_data(text_column='cleaned_text', label_column='label')
lr_model.train_model(X_train, y_train)
lr_model.evaluate_model(X_test, y_test)

# Save the final model
lr_model.save_model(filename='../saved_models/logistic_regression_model.joblib')


# Naive Bayes Model
print("\n###### Naive Bayes Model ######")
nb_model = NaiveBayesModel(df)
X_train, X_test, y_train, y_test = nb_model.preprocess_data(text_column='cleaned_text', label_column='label')
nb_model.train_model(X_train, y_train)
nb_model.evaluate_model(X_test, y_test)

# Save the final model
nb_model.save_model(filename='../saved_models/naive_bayes_model.joblib')

# SVM Model
print("\n###### Support Vector Machine ######")
svm_model = SVMModel(df)
X_train, X_test, y_train, y_test = svm_model.preprocess_data(text_column='cleaned_text', label_column='label')
svm_model.train_model(X_train, y_train)
svm_model.evaluate_model(X_test, y_test)

# Save the final model
nb_model.save_model(filename='../saved_models/svm_model.joblib')

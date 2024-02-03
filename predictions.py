import joblib

def load_and_predict(input_text):
    # Load Logistic Regression model
    loaded_lr_model = joblib.load('saved_models/logistic_regression_model.joblib')
    lr_prediction = loaded_lr_model.predict(input_text)

    # Load Naive Bayes model
    loaded_nb_model = joblib.load('saved_models/naive_bayes_model.joblib')
    nb_prediction = loaded_nb_model.predict(input_text)

    # Load SVM model
    loaded_svm_model = joblib.load('saved_models/svm_model.joblib')
    svm_prediction = loaded_svm_model.predict(input_text)

    return {
        'Logistic Regression Prediction': lr_prediction,
        'Naive Bayes Prediction': nb_prediction,
        'SVM Prediction': svm_prediction
    }

# Input_text
input_text = ["industrie"]
predictions = load_and_predict(input_text)

# Display predictions
for model, prediction in predictions.items():
    print(f"{model}: {prediction}")

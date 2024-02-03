import joblib
from fastapi import FastAPI

app = FastAPI()


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


@app.get("/")
def read_root():
    return "FASTAPI for DS_Coding_Challenge"


@app.get("/check_label/")
def get_prediction(text: str):
    input_text = [text]
    predictions = load_and_predict(input_text)
    prediction_result = {}
    for model, prediction in predictions.items():
        prediction_result[model] = prediction[0]

    return prediction_result

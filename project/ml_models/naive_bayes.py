import joblib
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from project.preprocessing.data_preprocessing import TextPreprocessor

class NaiveBayesModel:
    def __init__(self, dataframe):
        self.model = None
        self.df = dataframe
        self.text_preprocessor = TextPreprocessor(self.df)

    def preprocess_data(self, text_column='text', label_column='label'):
        # Drop rows with missing labels and text
        self.text_preprocessor.drop_rows_missing_labels(label_column=label_column)
        self.text_preprocessor.drop_rows_missing_labels(label_column=text_column)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = self.text_preprocessor.split_data(text_column=text_column,
                                                                             label_column=label_column)

        return x_train, x_test, y_train, y_test

    def train_model(self, x_train, y_train):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):
        # Make predictions on the test set
        y_pred = self.model.predict(x_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)

        # Display evaluation metrics
        print("\nAccuracy: {:.4f}".format(accuracy))
        print("\nClassification Report:")
        print(classification_rep)
        print("\nConfusion Matrix:")
        print(confusion_mat)

    def save_model(self, filename='naive_bayes_model.joblib'):  # Change the filename
        if self.model is None:
            raise ValueError("Model has not been trained. Please train the model first.")

        # Save the trained model to a file
        joblib.dump(self.model, filename)
        print(f"Model saved as {filename}")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    def __init__(self, dataframe):
        self.df = dataframe

    # Drop rows with missing labels
    def drop_rows_missing_labels(self, label_column='label'):
        self.df.dropna(subset=[label_column], inplace=True)

    # Converts text to Numerical Data
    @staticmethod
    def count_vectorize(text_train, text_test):
        vectorized = CountVectorizer()
        x_train = vectorized.fit_transform(text_train)
        x_test = vectorized.transform(text_test)
        return x_train, x_test

    @staticmethod
    def tfidf_vectorize(text_train, text_test):
        vectorized = TfidfVectorizer()
        x_train = vectorized.fit_transform(text_train)
        x_test = vectorized.transform(text_test)
        return x_train, x_test


    # Train Test Split
    def split_data(self, text_column='text', label_column='label'):
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(self.df[text_column], self.df[label_column], test_size=0.2,
                                                            random_state=42)

        return x_train, x_test, y_train, y_test

from project.ml_models.ml_main import df
import langid

"""
Overview:
The dataset's shape is 37,295 rows and 2 columns.
There are 100 missing values in the label column.
The average text length is around 22 characters, with a minimum of 1 character and a maximum of 798 characters.
The label distribution is imbalanced. Total 6 labels/classes.
All rows are not in the German language. However, majority of the non-German rows contain text that is in fact in German but getting classified as Non-German.
"""


def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang


class DataVisualization:
    def __init__(self, dataframe):
        self.df = dataframe

    def show_dataframe_shape(self):  # shape of dataframe
        print(self.df.shape)

    def show_missing_values(self):  # check for missing values
        missing_values = self.df.isnull().sum()
        print("Missing Values: \n", missing_values)

    def show_text_length_statistics(self):  # check the distribution of text lengths
        text_lengths = self.df['text'].apply(len)
        print("Text Length Statistics: \n", text_lengths.describe())

    def show_label_distribution(self):  # check the distribution of labels
        label_distribution = self.df['label'].value_counts()
        print("Label Distribution: \n", label_distribution)

    def apply_language_detection(self):  # apply the language detection function to each row in text
        self.df['detected_language'] = self.df['text'].apply(detect_language)

    def check_all_german(self):  # check if all detected languages are de
        all_german = all(self.df['detected_language'] == 'de')
        print("All rows are in German language:", all_german)

    def display_non_german_rows(self):  # display the rows with non-German text for manual inspection
        non_german_rows = self.df[self.df['detected_language'] != 'de']
        print("\nRows with non-German text:")
        print(non_german_rows[['text', 'detected_language']])


# Instantiate the class
dv = DataVisualization(df)

# method calls
dv.show_dataframe_shape()
dv.show_missing_values()
dv.show_text_length_statistics()
dv.show_label_distribution()
dv.apply_language_detection()
dv.check_all_german()
dv.display_non_german_rows()

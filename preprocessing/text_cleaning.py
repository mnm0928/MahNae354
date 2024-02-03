import re
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("german"))


def clean_text(text):
    """
    - Remove HTML tags
    - Keep only ASCII + European Chars (including umlauts ans ß) and whitespace
    - Remove single letter chars
    - Convert all whitespaces to single whitespace
    - Remove stopwords
    - Convert to lowercase
    """
    # Removing HTML Tags
    RE_TAGS = re.compile(r"<[^>]+>")
    text = re.sub(RE_TAGS, " ", text)

    # Keep only ASCII + European Chars and whitespace (including ö, ä, ü, ß)
    RE_NON_ALPHANUMERIC = re.compile(r"[^A-Za-zÀ-žöäüß ]", re.IGNORECASE)
    text = re.sub(RE_NON_ALPHANUMERIC, " ", text)

    # Remove single letter chars
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
    text = re.sub(RE_SINGLECHAR, " ", text)

    # Convert all whitespaces to single whitespace
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    text = re.sub(RE_WSPACE, " ", text)

    # Tokenize, lowercase, and remove stopwords
    word_tokens = word_tokenize(text)
    words_filtered = [word.lower() for word in word_tokens if word.lower() not in stop_words]

    text_clean = " ".join(words_filtered)
    return text_clean



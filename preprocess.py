"""
preprocess.py
Text cleaning and tokenisation utilities for sentiment analysis.
"""

import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    lowercase → strip HTML/URLs → remove non-alpha →
    remove stopwords → lemmatize.

    Parameters
    ----------
    text : str  Raw review text

    Returns
    -------
    str  Cleaned text ready for tokenisation
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)             # HTML tags
    text = re.sub(r'http\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'[^a-z\s]', '', text)          # non-alpha characters
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words]
    return ' '.join(tokens)


def batch_clean(texts: list) -> list:
    """Apply clean_text to a list of strings."""
    return [clean_text(t) for t in texts]


def tokenise_and_pad(texts: list, tokenizer, max_len: int = 100) -> np.ndarray:
    """
    Convert cleaned text to padded integer sequences.

    Parameters
    ----------
    texts     : list of str (already cleaned)
    tokenizer : fitted Keras Tokenizer
    max_len   : int  Padding length

    Returns
    -------
    np.ndarray, shape (N, max_len)
    """
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding='post')


def load_tokenizer(path: str = 'tokenizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_tokenizer(tokenizer, path: str = 'tokenizer.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f'Tokenizer saved to {path}')

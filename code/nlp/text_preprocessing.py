# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Standard Library Imports
import re

# Dependency Imports
import pandas as pd
from nltk.corpus import stopwords

# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #

def text_preprocessing(corpus):
    """
    Alters a list of document-length strings, modifying each string to remove
    redacted tokens, irregular whitespace, punctuation, English stop words, and
    tokens that are composed solely of numbers. All tokens are adjusted to be
    lowercased.

    Parameters:
    corpus (list): A list of document-length strings

    Returns:
    corpus (list): A list of nested lists containing the strings of individual
    tokens.
    """

    stop_words = set(stopwords.words('english'))

    for i, note in enumerate(corpus):
        note = [word for word in note.split(' ') if not re.match("(\[[\*]+[\w]+[\*]+\])", word)]
        note = ' '.join(note)
        note = re.sub(r'[^\w\s]', '', note)
        note = [word for word in note.split(' ') if not re.match("([\*]+)", word)]
        note = [word for word in note if word != '']
        note = [word.lower() for word in note]
        note = [word for word in note if word not in stop_words]
        corpus[i] = [word for word in note if not re.match("([0-9]+)", word)]

    return corpus

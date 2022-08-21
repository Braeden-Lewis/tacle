# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Standard Library Imports
import json
from pathlib import Path

# Dependency Imports
import nltk
import pandas as pd

# Local Imports
from . import feature_creation, text_preprocessing
from ..kit import timer

# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #

@timer
def execute(data_file):
    """Executes the natural language processing tasks of the TaCLE package."""

    with open("./../config.json", 'r') as jsonfile:
        config = json.load(jsonfile)

    nltk.download('stopwords')

    load_file = Path(str(config['directories']['EXTR_PICKLE_OUTPUT_DIR']) + ('/' + config['RUN_DATETIME'] + '-extr.pkl'))

    note_dataframe = pd.read_pickle(load_file)

    note_corpus = note_dataframe['content'].tolist()
    note_corpus = text_preprocessing.text_preprocessing(note_corpus)

    print(note_dataframe['--classification--'].value_counts())

    if refinement == "none":
        bow_dataframe = feature_creation.term_matrix(note_dataframe,
                                    note_corpus,
                                    ngram_range=config["NGRAM"],
                                    matrix_type=config["MATRIX_TYPE"])

    elif refinement == "shared":
        bow_dataframe = feature_creation.shared_term_matrix(note_dataframe,
                                           note_corpus,
                                           ngram_range=config["NGRAM"],
                                           matrix_type=config["MATRIX_TYPE"])
    else:
        raise Exception("Argument for matrix type is not valid.")

    save_file = Path(str(config["directories"]["NLP_PICKLE_OUTPUT_DIR"]) + ('/' + config["RUN_DATETIME"] + '-nlp.pkl'))
    bow_dataframe.to_pickle(save_file)

    time.sleep(2)

# -------------------------------------------------------------------------- #
# ---------------------------- EXECUTABLES --------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    execute(refinement=config["REFINEMENT"])

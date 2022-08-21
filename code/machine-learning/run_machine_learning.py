# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Standard Library Imports
import json
from pathlib import Path

# Dependency Imports
import pandas as pd

# Local Imports
from . import model_evaluation
from .. kit import timer
# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #

@timer
def execute(matrix_type: str):
    """Executes the machine learning tasks of the TaCLE package."""

    with open("./../config.json", 'r') as jsonfile:
        config = json.load(jsonfile)

    load_file = Path(str(config['directories']['NLP_PICKLE_OUTPUT_DIR']) + ('/' + config['RUN_DATETIME'] + '-nlp.pkl'))

    bow_dataframe = pd.read_pickle(load_file)

    X = bow_dataframe.iloc[:, 0:bow_dataframe.shape[1]-1]
    y = bow_dataframe.iloc[:, bow_dataframe.shape[1]-1]

    model_evaluation.model_decision_tree(X, y)
    model_evaluation.model_knn(X, y)
    model_evaluation.model_logistic_regression(X, y)
    model_evaluation.model_svm(X, y)
    model_evaluation.model_xgboost(X, y)

    time.sleep(2)
# -------------------------------------------------------------------------- #
# ---------------------------- EXECUTABLES --------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    execute(matrix_type=config["MATRIX_TYPE"])

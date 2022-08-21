# ---------------------------------------------------------------------------- #
# --------------------------------- IMPORTS ---------------------------------- #
# ---------------------------------------------------------------------------- #
# Standard Library Imports
import json
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------- #
# ------------------------------- EXECUTABLES -------------------------------- #
# ---------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))

configurations = {
    "directories" : {
        "ROOT" : ROOT,
        "V0_DATA_IMPORT_DIR" : Path(ROOT + "data/input/v0/"),
        "EXTR_PICKLE_OUTPUT_DIR" : Path(ROOT + "data/output/pickle-files/extraction/"),
        "NLP_PICKLE_OUTPUT_DIR" : Path(ROOT + "data/output/pickle-files/nlp/"),
        "ML_MAT_OUTPUT_DIR" : Path(ROOT + "data/output/pickle-files/mach-learning/"),
    },
    "model-hyperparameters" : {
        "DECISION_TREE_PARAMETERS": {
            "criterion" : "gini",
            "splitter" : "best",
            "max_depth" : 10,
            "max_features" : "sqrt",
            "min_samples_split" : 3,
            "min_samples_leaf" : 4,
            "class_weight" : "balanced"
        },
        "LOG_REG_PARAMETERS" : {
            "penalty" : "l2",
            "C" : 11.288378916846883,
            "solver" : "liblinear",
            "max_iter": 500
        },
        "SVM_PARAMETERS" : {
            "kernel" : "linear",
            "C" : 0.0018329807108324356,
            "degree" : 0,
            "gamma" : "scale",
            "cache_size" : 100,
            "class_weight" : "balanced"
        },
        "KNN_PARAMETERS" : {
            "n_neighbors": 20,
            "weights" : "distance",
            "algorithm" : "auto",
            "leaf_size" : 10,
            "p" : 2,
            "n_jobs" : -1
        },
         "XGB_PARAMETERS" : {
            "learning_rate" : 0.1,
            "max_depth" : 2,
            "min_child_weight" : 2,
            "n_estimators" : 200,
            "nthread" : 1,
            "objective" : "binary:logistic", #"multi:softmax", # use "binary:logistic" for binary tasks, "multi:softmax" for multiclass
            "gamma" : 0.2,
            "subsample" : 0.3,
            "colsample_bytree" : 0.2,
            "eval_metric" : "mlogloss",
            "use_label_encoder" : False
        }
    },
    "parameter-tuning" : {
        "DECISION_TREE_TUNING" : {
            "criterion" : ["gini", "entropy"],
            "splitter" : ["best", "random"],
            "max_depth" : [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70],
            "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16],
            "min_samples_leaf" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "max_features" : ["auto", "sqrt"],
            "class_weight" : ["balanced"]
        },
        "LOG_REG_TUNING" : [
            {"penalty": ["l1", "l2"],
             "C" : np.logspace(-4, 4, 20),
             "solver": ["liblinear", "saga"],
             "max_iter" : [500, 1000, 2000, 5000]},
            {"penalty" : ["l2"],
            "C" : np.logspace(-4, 4, 20),
            "solver" : ["lbfgs", "newton-cg", "sag"],
            "max_iter" : [500, 1000, 2000, 5000]}
        ],
        "SVM_TUNING" : {
            "kernel" : ["linear", "poly", "rbf", "sigmoid"],
            "C" : np.logspace(-4, 4, 20),
            "degree" : [0, 1, 2, 3, 4, 5],
            "gamma" : ["scale", "auto"],
            "cache_size" : [100, 200, 300, 400, 500],
            "class_weight": ["balanced"]
        },
        "KNN_TUNING" : {
            "n_neighbors" : [5, 10, 15, 20, 25, 30, 35, 40],
            "weights" : ["uniform", "distance"],
            "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size" : [10, 20, 30, 40, 50, 60],
            "p" : [1, 2],
            "n_jobs" : [-1]
        },
        "XGB_TUNING" : {
            "max_depth" : [2, 3, 4, 5],
            "min_child_weight" : [2, 3, 4, 5],
            "n_estimators" : [50, 100, 150, 200],
            "nthread" : [1, 2, 3],
            "objective" : ["binary:logistic"], # binary:logistic for binary tasks, multi:softmax for multiclass
            "gamma" : [0.1, 0.2, 0.3],
            "subsample" : [0.1, 0.2, 0.3],
            "colsample_bytree" : [0.1, 0.2, 0.3],
            "eval_metric" : ["mlogloss"],
            "use_label_encoder" : [False]
        }
    },
    "RUN_DATETIME" : datetime.now().strftime("%Y%m%d-%H%M%S"),
    "NGRAM" : (1, 3),
    "DETECTABLE_CLASSES": ["BREAST", "BOTTLE", "EXPRESS/PUMP", "NA"],
    "CONCAT_CLASS" : {"FEEDING": ["BREAST", "BOTTLE", "EXPRESS/PUMP"]},
    "MATRIX_TYPE" : "tf-idf", # Can be "tf-idf" or "count"
    "REFINEMENT" : "none", # Can be "shared" or "none" (default)
    "MIN_DOC_FREQ": 30,
    "TEST_SIZE" : 0.20,
    "VALIDATION_SIZE" : 0.25,
    "CROSS_VALIDATIONS": 5,
    "RANDOM_STATE" : 22
}


if __name__ == "__main__":
    with open("config.json", "w+") as jsonfile:
        json.dump(configurations, jsonfile)

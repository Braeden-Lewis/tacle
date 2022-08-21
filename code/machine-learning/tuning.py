# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Standard Library Imports
import json

# Dependency Imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

# Local Imports
from .. import kit.timer as timer
# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #
def hyperparameter_tuning(X, y, model, param_grid):
    """
    """
    int_encoder = {value: i for i, value in enumerate(y.unique().tolist())}

    y = y.replace(y.unique().tolist(), list(range(len(y.unique()))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"])

    gscv = GridSearchCV(model,
                        param_grid=param_grid,
                        scoring="f1_macro",
                        cv=config["CROSS_VALIDATIONS"],
                        verbose=True,
                        n_jobs=-1,
                        error_score="raise")

    gscv.fit(X_train, y_train)

    print(gscv.best_params_)
    print(gscv.best_score_)
    print(gscv.best_estimator_)


def feature_selection(X, y, model):
    """
    """
    int_encoder = {value: i for i, value in enumerate(y.unique().tolist())}

    y = y.replace(y.unique().tolist(), list(range(len(y.unique()))))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"])

    model.fit(X_train, y_train)

    columns = []
    feature_importance = []

    for i, column in enumerate(X_train):
        columns.append(column)
        feature_importance.append(model.feature_importances_[i])

    feature_dataframe = zip(columns, feature_importance)
    feature_dataframe = pd.DataFrame(feature_dataframe, columns=["features", "feature_importance"])

    feature_dataframe = feature_dataframe.sort_values("feature_importance", ascending=False)

    for row in feature_dataframe.itertuples():
        if row[2] > 0:
            print("Feature: {} \t Importance: {}".format(row[1], row[2]))

@timer
def execute(model, matrix_type: str):
    """Executes the model tuning tasks of the TaCLE package."""

    with open("./../config.json", 'r') as jsonfile:
        config = json.load(jsonfile)

    load_file = Path(str(config['directories']['NLP_PICKLE_OUTPUT_DIR']) + ('/' + config['RUN_DATETIME'] + '-nlp.pkl'))

    bow_dataframe = pd.read_pickle(load_file)

    X = bow_dataframe.iloc[:, 0:bow_dataframe.shape[1]-1]
    y = bow_dataframe.iloc[:, bow_dataframe.shape[1]-1]

    print(type(model).__name__)
    hyperparameter_tuning(X, y, model, matrix_type)
    print("----------\n")
# -------------------------------------------------------------------------- #
# ---------------------------- EXECUTABLES --------------------------------- #
# -------------------------------------------------------------------------- #
# Executables needs to be updated to use the execute function for each model.
if __name__ == "__main__":
    with open("./../config.json", 'r') as jsonfile:
        config = json.load(jsonfile)

    load_file = Path(str(config['directories']['NLP_PICKLE_OUTPUT_DIR']) + ('/' + config['RUN_DATETIME'] + '-nlp.pkl'))

    bow_dataframe = pd.read_pickle(load_file)

    X = bow_dataframe.iloc[:, 0:bow_dataframe.shape[1]-1]
    y = bow_dataframe.iloc[:, bow_dataframe.shape[1]-1]

    print("DecisionTree")
    hyperparameter_tuning(X, y, model=DecisionTreeClassifier(), param_grid=config["parameter-tuning"]["DECISION_TREE_TUNING"])
    print("----------\n")

    print("KNeighbors")
    hyperparameter_tuning(X, y, model=KNeighborsClassifier(), param_grid=config["parameter-tuning"]["KNN_TUNING"])
    print("----------\n")

    print("LogisticRegression")
    hyperparameter_tuning(X, y, model=LogisticRegression(), param_grid=config["parameter-tuning"]["LOG_REG_TUNING"])
    print("----------\n")

    print("SVC")
    hyperparameter_tuning(X, y, model=SVC(), param_grid=config["parameter-tuning"]["SVM_TUNING"])
    print("----------\n")

    print("XGBoost")
    hyperparameter_tuning(X, y, model=XGBClassifier(use_label_encoder=False), param_grid=config["parameter-tuning"]["XGB_TUNING"])
    print("----------\n")

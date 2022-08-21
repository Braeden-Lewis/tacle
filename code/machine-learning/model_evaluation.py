# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Standard Library Imports
import os
import pickle

# Dependency Imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #

def model_decision_tree(X, y):
    """
    """
    model = DecisionTreeClassifier(criterion=config["model-hyperparameters"]["DECISION_TREE_PARAMETERS"]["criterion"],
                                   splitter=config["model-hyperparameters"]["DECISION_TREE_PARAMETERS"]["splitter"],
                                   max_depth=config["model-hyperparameters"]["DECISION_TREE_PARAMETERS"]["max_depth"],
                                   max_features=config["model-hyperparameters"]["DECISION_TREE_PARAMETERS"]["max_features"],
                                   min_samples_split=config["model-hyperparameters"]["DECISION_TREE_PARAMETERS"]["min_samples_split"],
                                   min_samples_leaf=config["model-hyperparameters"]["DECISION_TREE_PARAMETERS"]["min_samples_leaf"],
                                   class_weight=config["model-hyperparameters"]["DECISION_TREE_PARAMETERS"]["class_weight"],
                                   random_state=config["RANDOM_STATE"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"])

    model.fit(X_train, y_train)

    report = classification_report(y_test.tolist(), model.predict(X_test).tolist(), output_dict=True)

    report_dataframe = pd.DataFrame(report).transpose()

    model_save_file = os.path.join(config["directories"]["MLMODEL_PICKLE_OUTPUT_DIR"],
                                   config["RUN_DATETIME"] + "-" + str(type(model).__name__) + ".pkl")

    csv_save_file = os.path.join(config["directories"]["DESC_STATS_OUTPUT_DIR"],
                                 config['RUN_DATETIME'] + "-" + str(type(model).__name__) + "-stats.csv")

    pickle.dump(model, open(model_save_file, "wb"))
    report_dataframe.to_csv(csv_save_file)


def model_logistic_regression(X, y):
    """
    """
    model = LogisticRegression(penalty=config["model-hyperparameters"]["LOG_REG_PARAMETERS"]["penalty"],
                               C=config["model-hyperparameters"]["LOG_REG_PARAMETERS"]["C"],
                               solver=config["model-hyperparameters"]["LOG_REG_PARAMETERS"]["solver"],
                               max_iter=config["model-hyperparameters"]["LOG_REG_PARAMETERS"]["max_iter"],
                               random_state=config["RANDOM_STATE"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"])

    model.fit(X_train, y_train)

    report = classification_report(y_test.tolist(), model.predict(X_test).tolist(), output_dict=True)

    report_dataframe = pd.DataFrame(report).transpose()

    model_save_file = os.path.join(config["directories"]["MLMODEL_PICKLE_OUTPUT_DIR"],
                                   config["RUN_DATETIME"] + "-" + str(type(model).__name__) + ".pkl")

    csv_save_file = os.path.join(config["directories"]["DESC_STATS_OUTPUT_DIR"],
                                 config['RUN_DATETIME'] + "-" + str(type(model).__name__) + "-stats.csv")

    pickle.dump(model, open(model_save_file, "wb"))
    report_dataframe.to_csv(csv_save_file)


def model_svm (X, y):
    """
    """
    model = SVC(kernel=config["model-hyperparameters"]["SVM_PARAMETERS"]["kernel"],
                C=config["model-hyperparameters"]["SVM_PARAMETERS"]["C"],
                degree=config["model-hyperparameters"]["SVM_PARAMETERS"]["degree"],
                gamma=config["model-hyperparameters"]["SVM_PARAMETERS"]["gamma"],
                cache_size=config["model-hyperparameters"]["SVM_PARAMETERS"]["cache_size"],
                class_weight=config["model-hyperparameters"]["SVM_PARAMETERS"]["class_weight"],
                random_state=config["RANDOM_STATE"])

    X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"])

    model.fit(X_train, y_train)

    report = classification_report(y_test.tolist(), model.predict(X_test).tolist(), output_dict=True)

    report_dataframe = pd.DataFrame(report).transpose()

    model_save_file = os.path.join(config["directories"]["MLMODEL_PICKLE_OUTPUT_DIR"],
                                   config["RUN_DATETIME"] + "-" + str(type(model).__name__) + ".pkl")

    csv_save_file = os.path.join(config["directories"]["DESC_STATS_OUTPUT_DIR"],
                                 config['RUN_DATETIME'] + "-" + str(type(model).__name__) + "-stats.csv")

    pickle.dump(model, open(model_save_file, "wb"))
    report_dataframe.to_csv(csv_save_file)

def model_knn(X, y):
    """
    """
    model = KNeighborsClassifier(n_neighbors=config["model-hyperparameters"]["KNN_PARAMETERS"]["n_neighbors"],
                                 weights=config["model-hyperparameters"]["KNN_PARAMETERS"]["weights"],
                                 algorithm=config["model-hyperparameters"]["KNN_PARAMETERS"]["algorithm"],
                                 leaf_size=config["model-hyperparameters"]["KNN_PARAMETERS"]["leaf_size"],
                                 p=config["model-hyperparameters"]["KNN_PARAMETERS"]["p"],
                                 n_jobs=config["model-hyperparameters"]["KNN_PARAMETERS"]["n_jobs"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"])

    model.fit(X_train, y_train)

    report = classification_report(y_test.tolist(), model.predict(X_test).tolist(), output_dict=True)

    report_dataframe = pd.DataFrame(report).transpose()

    model_save_file = os.path.join(config["directories"]["MLMODEL_PICKLE_OUTPUT_DIR"],
                                   config["RUN_DATETIME"] + "-" + str(type(model).__name__) + ".pkl")

    csv_save_file = os.path.join(config["directories"]["DESC_STATS_OUTPUT_DIR"],
                                 config['RUN_DATETIME'] + "-" + str(type(model).__name__) + "-stats.csv")

    pickle.dump(model, open(model_save_file, "wb"))
    report_dataframe.to_csv(csv_save_file)

def model_xgboost(X, y):
    """
    """
    model = XGBClassifier(learning_rate=config["model-hyperparameters"]["XGB_PARAMETERS"]["learning_rate"],
                          max_depth=config["model-hyperparameters"]["XGB_PARAMETERS"]["max_depth"],
                          min_child_weight=config["model-hyperparameters"]["XGB_PARAMETERS"]["min_child_weight"],
                          n_estimators=config["model-hyperparameters"]["XGB_PARAMETERS"]["n_estimators"],
                          nthread=config["model-hyperparameters"]["XGB_PARAMETERS"]["nthread"],
                          objective=config["model-hyperparameters"]["XGB_PARAMETERS"]["objective"],
                          gamma=config["model-hyperparameters"]["XGB_PARAMETERS"]["gamma"],
                          subsample=config["model-hyperparameters"]["XGB_PARAMETERS"]["subsample"],
                          colsample_bytree=config["model-hyperparameters"]["XGB_PARAMETERS"]["colsample_bytree"],
                          eval_metric=config["model-hyperparameters"]["XGB_PARAMETERS"]["eval_metric"],
                          use_label_encoder=config["model-hyperparameters"]["XGB_PARAMETERS"]["use_label_encoder"],
                          random_state=model_hyperparameters["RANDOM_STATE"])

    int_decoder = {str(i): value for i, value in enumerate(y.unique().tolist())}

    y = y.replace(y.unique().tolist(), list(range(len(y.unique()))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_STATE"])

    model.fit(X_train, y_train)


    report = classification_report(y_test.tolist(), model.predict(X_test).tolist(), output_dict=True)
    report = {(int_decoder[key] if key in int_decoder else key):value for key, value in report.items()}

    report_dataframe = pd.DataFrame(report).transpose()

    model_save_file = os.path.join(config["directories"]["MLMODEL_PICKLE_OUTPUT_DIR"],
                                   config["RUN_DATETIME"] + "-" + str(type(model).__name__) + ".pkl")

    csv_save_file = os.path.join(config["directories"]["DESC_STATS_OUTPUT_DIR"],
                                 config['RUN_DATETIME'] + "-" + str(type(model).__name__) + "-stats.csv")

    pickle.dump(model, open(model_save_file, "wb"))
    report_dataframe.to_csv(csv_save_file)

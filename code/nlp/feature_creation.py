# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Dependency Imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #
def term_matrix(dataframe, corpus, ngram_range, matrix_type):
    """
    """
    for i, note in enumerate(corpus):
        corpus[i] = ' '.join(note)

    dataframe.drop(dataframe.columns[1], axis=1, inplace=True)
    dataframe['content'] = corpus

    if matrix_type == "tf-idf":
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=config["MIN_DOC_FREQ"])
    elif matrix_type == "count":
        vectorizer = CountVectorizer(analyzer='word', ngram_range=ngram_range, min_df=config["MIN_DOC_FREQ"])
    else:
        raise Exception("Argument for matrix type must be either \'tf-idf\' or \'count\'")

    X = vectorizer.fit_transform(dataframe['content'])

    bow_dataframe = pd.DataFrame(X.toarray(),
                                 columns=vectorizer.get_feature_names_out(),
                                 index=dataframe.index)

    bow_dataframe['--classification--'] = dataframe['--classification--']

    return bow_dataframe


def shared_term_matrix(dataframe, corpus, ngram_range, matrix_type):
    """
    """

    for i, note in enumerate(corpus):
        corpus[i] = ' '.join(note)

    dataframe.drop(dataframe.columns[1], axis=1, inplace=True)
    dataframe['content'] = corpus

    def _split_dataframe(dataframe, detectable_classes: list):
        """
        Divides a single pandas dataframe into a list of dataframes, segregated by
        manually annotated response variables in string format.

        Parameters:
        dataframe (pandas.DataFrame): A dataframe containing a column of
        string-represented classifiers for the data.

        concat_class (bool): Used to concatenate the data of two similar classifiers
        into a single classifier.

        Returns:
        dataframe_list (list): A list containing dataframes separated by
        classification
        """

        dataframe_list = []
        for elem in dataframe["--classification--"].unique():
            df = pd.DataFrame([row for row in dataframe.itertuples() if row[1] == elem],
                              columns=["Index", "--classification--", "content"])
            dataframe_list.append(df)

        return dataframe_list

    separated_dataframe = _split_dataframe(dataframe,
                                          detectable_classes=config["DETECTABLE_CLASSES"])

    if matrix_type == "tf-idf":
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=config["MIN_DOC_FREQ"])
    elif matrix_type == "count":
        vectorizer = CountVectorizer(analyzer='word', ngram_range=ngram_range, min_df=config["MIN_DOC_FREQ"])
    else:
        raise Exception("Argument for matrix type must be either \'tf-idf\' or \'count\'")

    bow_dataframe_list = []
    for df in separated_dataframe:
        X = vectorizer.fit_transform(df['content'])
        bow_dataframe = pd.DataFrame(X.toarray(),
                                     columns=vectorizer.get_feature_names_out(),
                                     index=df.index)

        bow_dataframe['--classification--'] = df['--classification--']
        bow_dataframe_list.append(bow_dataframe)

    features = bow_dataframe_list[0].columns

    for i in range(len(bow_dataframe_list)):
        features = np.intersect1d(features, bow_dataframe_list[i].columns)
    features = features.tolist()

    output_dataframe = bow_dataframe_list[0][features]

    for i in range(len(bow_dataframe_list)-1):
        output_dataframe = pd.merge(output_dataframe[features], bow_dataframe_list[i+1][features], how='outer')

    return output_dataframe

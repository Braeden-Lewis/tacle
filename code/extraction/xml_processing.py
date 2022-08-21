# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Standard Library Imports
import os
import re
import xml.etree.ElementTree as et
from pathlib import Path

# Dependency Imports
import pandas as pd

# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #

def xml_extraction(data_directory:str, concat_class:dict):
    """
    Loops through all annotated xml documents to collect values for baby
    and note numbers, annotated classification, and text content. Creates two
    dictionaries, both using the same keys.

    Parameters:
    data_directory (string): A string denoting the path where xml files are
    located.

    concat_class (dict): Used to concatenate the data of two similar classifiers
    into a single classifier.

    Returns:
    class_dict (dict): A dictionary with keys that are tuples containing
    patient ID and note number, and values that are the manually annotated
    classification of the respective clinical note in string format.

    text_dict (dict): A dictionary with keys that are tuples containing
    patient ID and note number, and values that are the contents of the
    clinical notes themselves in string format.
    """

    class_dict = {}
    text_dict = {}

    for root, dirs, files in os.walk(data_directory):
        for file in sorted(files):
            if file.endswith('.xml'):
                etree = et.parse(os.path.join(root, file))
                baby_note = etree.findall('document//passage')[-1].find("text").text
                baby_note = tuple(re.findall('[\d]+', baby_note))
                text_content = etree.findall('document//passage')[0].find("text").text

                if baby_note not in class_dict.keys():
                    class_dict[baby_note] = set()
                if baby_note not in text_dict.keys():
                    text_dict[baby_note] = text_content

                annotations = etree.findall('document//annotation')

                for annotation in annotations:
                    annotation_type = annotation.find("infon[@key='type']").text
                    annotation_text = annotation.find("text").text
                    if annotation_type == 'FEED_CLASS':
                        class_dict[baby_note].add(annotation_text)

    class_dict = {key:element for key, value in class_dict.items() if len(value) == 1 for element in value}

    def _class_concatination(class_dict, concat_class):
        """
        Adjusts the manual annotated classifications to merge any labels as
        defined within the config.json file under the key CONCAT_CLASS.

        Parameters:
        class_dict (dict): A dictionary with keys that are tuples containing
        patient ID and note number, and values that are the manually annotated
        classification prior to any adjustments in classifications.

        concat_class (dict): Used to concatenate the data of two similar
        classifiers into a single classifier.

        Returns:
        class_dict (dict): the resulting class_dict with classifications
        adjusted.
        """
        for k, v in concat_class.items():
            for elem in v:
                class_dict = {key:(k if value == elem else value) for key, value in class_dict.items()}

        class_dict = {key: value for key, value in class_dict.items() if value in config["DETECTABLE_CLASSES"] or value in config["CONCAT_CLASS"]}
        return class_dict

    if concat_class:
        class_dict = _class_concatination(class_dict, concat_class)

    return class_dict, text_dict


def structure_dataframe(class_dict:dict, text_dict:dict):
    """
    Takes two dictionaries that share keys, but with different values, and
    creates a pandas dataframe. Keys are set as the dataframe's index.

    Parameters:
    class_dict (dict): dictionary that containes key:value with a tuple of
    patient ID and note number as the key and the manually annotated
    classification, as a string, for the respective note as the value.

    text_dict (dict): dictionary that containes key:value with a tuple of
    patient ID and note number as the key and the contents of the note in string
    format as the value.

    Returns:
    dataframe (pandas.DataFrame): A dataframe that combines the two dictionaries.
    """

    data = [class_dict, text_dict]
    data = {key:[d[key] for d in data] for key in data[0]}
    dataframe = pd.DataFrame.from_dict(data,
                                orient='index',
                                columns=['classification', 'content'])
    dataframe.index.name='id_note'
    return dataframe

# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Standard Library Imports
import json
from time import time

# Local Imports
from . import xml_processing
from .. import kit

# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #

@timer
def execute():
    """Executes the extraction process of the TaCLE package."""
    with open("./../config.json", 'r') as jsonfile:
        config = json.load(jsonfile)

    class_dict, text_dict = xml_processing.xml_extraction(config["directories"]["V0_DATA_IMPORT_DIR"],
                                                          concat_class=config['CONCAT_CLASS'])

    note_dataframe = xml_processing.structure_dataframe(class_dict, text_dict)

    save_file = Path(str(config['directories']['EXTR_PICKLE_OUTPUT_DIR']) + ('/' + config['RUN_DATETIME'] + '-extr.pkl'))
    note_dataframe.to_pickle(save_file)

    time.sleep(2)
# -------------------------------------------------------------------------- #
# ---------------------------- EXECUTABLES --------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    execute()

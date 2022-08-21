# -------------------------------------------------------------------------- #
# ------------------------------ IMPORTS ----------------------------------- #
# -------------------------------------------------------------------------- #
# Standard Library Imports
import os
from time import time

# -------------------------------------------------------------------------- #
# ----------------------------- FUNCTIONS ---------------------------------- #
# -------------------------------------------------------------------------- #

def timer(method):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        x = method(*args, **kwargs)
        end_time = time.perf_counter()
        time_elapsed = end_time - start_time
        print("Run time (seconds): ", time_elapsed)
        return x
    return wrapper

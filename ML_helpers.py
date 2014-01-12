
import numpy
import random

""" Helper functions for machine-learning. This needs to be added to workflow/util.py
    if these scripts are added to qiime.
"""

def bool_cast(s):
    if s.lower() == 'true' or s.lower() == 't':
        return True
    elif s.lower() == 'false' or s.lower() == 'f':
        return False

    raise ValueError

def num_cast(s):
    if float(s) % 1 == 0:
        return int(s)
    return float(s)

def cast(s):
    for cast_func in (num_cast, bool_cast, str):
        try:
            return cast_func(s)
        except ValueError:
            pass

    raise BaseException('Should not reach here... the input parameter value ' \
            'can\'t be cast as a string?')


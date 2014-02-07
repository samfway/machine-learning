#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Dan Malmer", "Will Van Treuren", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.7.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Machine Learning utility script 
"""

def bool_cast(s):
    """ Cast string to boolean """
    if s.lower() == 'true' or s.lower() == 't':
        return True
    elif s.lower() == 'false' or s.lower() == 'f':
        return False
    raise ValueError('Could not cast to ')

def num_cast(s):
    """ Convert string to int/float """ 
    if float(s) % 1 == 0:
        return int(s)
    return float(s)

def custom_cast(s):
    """ Convert to number/binary/string in that order of preference """
    for cast_func in (num_cast, bool_cast, str):
        try:
            return cast_func(s)
        except ValueError:
            pass
    raise BaseException('Could not cast as number/string!')

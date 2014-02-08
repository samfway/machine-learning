#!/usr/bin/env python
from numpy import array
import argparse
from argparse import RawTextHelpFormatter
import ml_lib.parse as ml_parse

def interface():
    args = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description='Print predictions to screen.',
        epilog='')
    args.add_argument('-i', '--prediction-file', help='Predictions file (pickle)', required=True)
    args = args.parse_args()
    return args

def view_predictions(prediction_file):
    """ Print predictions to screen """ 
    predictions = ml_parse.load_predictions_from_file(prediction_file)
    for model in predictions.keys():
        print 'MODEL: %s' % (model)
        for predicted, actual in predictions[model]:
            for p, a in zip(predicted, actual):
                print '%s (%s)' % (str(p), str(a))
        print ''

if __name__=="__main__":
    args = interface()
    view_predictions(args.prediction_file)


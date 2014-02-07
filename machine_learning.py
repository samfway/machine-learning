#!/usr/bin/env python
from numpy import array
import argparse
from argparse import RawTextHelpFormatter
import ml_lib.parse as ml_parse
import ml_lib.evaluation as ml_eval
import ml_lib.machine_learning as ml

def interface():
    args = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description='Simple wrapper for machine_learning.py.',
        epilog='Supply the following:\n' + \
                  '* OTU table (.biom)\n\n' + \
                  'And either of these two options:\n' + \
                  '1) Mapping file + Metadata category + (optional) Metadata value\n' + \
                  '2) Separate labels file')
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    args.add_argument('-m', '--mapping-file', help='Mapping table')
    args.add_argument('-l', '--labels-file', help='Labels file')
    args.add_argument('-p', '--prediction-file', help='Predictions dictionary (pickle)', default='pred.pkl')
    args.add_argument('-c', '--metadata-category', help='Metadata category')
    args.add_argument('-v', '--metadata-value', help='Metadata value') 
    args.add_argument('--dm', action='store_true', help='Input ' + \
        'matrix is a distance matrix', default=False)
    args.add_argument('-s', '--sklearn-file', help='Scikit-learn configuration file')
    args.add_argument('-f', '--find-features', help='Find best features for each classifier', \
                            action='store_true')
    args.add_argument('-o', '--output-file', help='Output file for report (default: ML-report.txt)', \
                            default='ML-report.txt', type=str)
    args = args.parse_args()
    return args

def evaluate_classifiers(list_of_models, model_names, data_matrix, actual_values, find_features, \
                        output_file, is_distance_matrix=False):
    """ Run and evaluate the supplied dataset using 10-fold cross-validation """
    test_sets = ml.get_test_sets(actual_values, 10)
    predictions, timers = ml.get_cross_validation_results(list_of_models, model_names, data_matrix, \
                    actual_values, test_sets, find_features, is_distance_matrix)
    ml_eval.evaluate_classification_results(model_names, predictions, list(set(actual_values)), timers, output_file)
    return predictions 

if __name__=="__main__":
    args = interface()
    data_matrix, sample_ids, actual_values = ml_parse.load_dataset(args.data_matrix, args.mapping_file, \
        args.metadata_category, args.metadata_value, args.labels_file, args.dm)
    list_of_models, model_names = ml.build_list_of_classifiers(args.sklearn_file)
    predictions = evaluate_classifiers(list_of_models, model_names, data_matrix, actual_values, \
                                        args.find_features, args.output_file, args.dm)
    ml.save_predictions_to_file(predictions, args.prediction_file)


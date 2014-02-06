#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Dan Malmer", "Will Van Treuren", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.7.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Evaluation functions 
"""

from numpy import array
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

def evaluate_classification_results(model_names, predictions, possible_labels, timers, output_file):
    """ Classification report. 
        "predictions" is a dictionary, with each key being the name of a classifier. 
        The value assigned to each dictionary key is a list of tuples.  Each tuple contains a list
        of predicted and true labels for a classification task.  

        This function evalutates each classifier and prints out an average score 
        for precision, recall, and f1.  
    """
    
    # Label dict will be used to convert labels into integer values for interfacing with 
    # scikit-learn's evaluation scripts. 
    label_dict = { key:value for key, value in zip(possible_labels, range(len(possible_labels))) }
    label_length = max([len(label) for label in model_names])

    f_out = open(output_file, 'w')
    f_out.write('Scores by classifier\n')
    f_out.write('--------------------\n\n')

    for classifier_name in model_names:

        f_out.write('%s (total time = %.2fs):\n' % (classifier_name, timers[classifier_name]))

        accuracy_scores = []
        precision_scores = []
        recall_scores = [] 
        f1_scores = [] 
    
        for predicted, actual in predictions[classifier_name]:
            predicted  = [label_dict[p] for p in predicted] 
            actual = [label_dict[a] for a in actual] 
            accuracy_scores.append(accuracy_score(actual, predicted))
            precision_scores.append(precision_score(actual, predicted))
            recall_scores.append(recall_score(actual, predicted))
    
        accuracy_scores = array(accuracy_scores)
        precision_scores = array(precision_scores)
        recall_scores = array(recall_scores)
        f1_scores = array([ (2*p*r)/(p+r) for p,r in \
                zip(precision_scores, recall_scores) ])
        a_mean, a_dev = accuracy_scores.mean(), 2*accuracy_scores.std()
        p_mean, p_dev = precision_scores.mean(), 2*precision_scores.std()
        r_mean, r_dev = recall_scores.mean(), 2*recall_scores.std()
        f_mean, f_dev = f1_scores.mean(), 2*f1_scores.std()
    
        f_out.write(' Mean accuracy = %.3f (+/- %.2f)\n' % (a_mean, a_dev))
        f_out.write(' Mean precision = %.3f (+/- %.2f)\n' % (p_mean, p_dev))
        f_out.write(' Mean recall = %.3f (+/- %.2f)\n' % (r_mean, r_dev))
        f_out.write(' Mean f-score = %.3f (+/- %.2f)\n' % (f_mean, f_dev))
        f_out.write('\n')

    f_out.close()

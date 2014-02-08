#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Dan Malmer", "Will Van Treuren", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.7.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Machine Learning playground script 
"""

from collections import defaultdict
from numpy import asarray, array
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from time import clock
from util import custom_cast
import matplotlib.pyplot as plt
import warnings

def get_test_sets(class_labels, kfold=10, stratified=True):
    """ Generate lists of indices for performing k-fold cross validation
        Note: test sets are stratefied, meaning that the number of class labels
        in each test set is proportional to the number in the whole set. 
    """
    if stratified: 
        return StratifiedKFold(class_labels, kfold)
    else: 
        return KFold(len(class_labels), kfold)

def get_predictions(model, training_matrix, training_values, test_matrix):
    """ Train and run and classification/regression model.  
        Returns: predicted values for each element in the test set. 
    """ 
    model.fit(training_matrix, training_values)
    predictions = model.predict(test_matrix) 
    return predictions

def get_test_train_set(data_matrix, actual_values, test_set, is_distance_matrix=False):
    """ Create test/train data set for cross-validation 
        Inputs:
            + "data_matrix": sample/feature matrix 
            + "actual_values": labels for each of the samples in data_matrix    
            + "test_set": index lists for training and testing examples 
            + "is_distance_matrix": data_matrix is a distance matrix. 
    """

    train_idx, test_idx = test_set

    if not is_distance_matrix:
        train_matrix = data_matrix[train_idx,:]
        train_values = actual_values[train_idx]
        test_matrix = data_matrix[test_idx,:]
        test_values = actual_values[test_idx]
    else:
        train_matrix = data_matrix[train_idx,:][:,train_idx]
        train_values = actual_values[train_idx]
        test_matrix = data_matrix[test_idx,:][:,train_idx]
        test_values = actual_values[test_idx]

    return train_matrix, train_values, test_matrix, test_values 

def get_cross_validation_results(list_of_models, model_names, data_matrix, actual_values, test_sets, \
                                find_features, is_distance_matrix=False):
    """ Get prediction results for a given list of models, performing cross-validation with the supplied test sets
        Inputs:
            + "list_of_models": model objects that implement .fit() and .predict() 
            + "model_names": name/identifier describing each model (i.e. "SVM", "Random Forest", etc.)
            + "data_matrix": sample/feature matrix for the dataset.  The index sets in test_sets divide this 
               into cross validation matrices.  
            + "actual_values": known values/labels for each sample in the matrix. 
            + "test_sets": index lists dividing the data_matrix into test and train matrices for cross validation. 
    """

    predictions = defaultdict(list)
    timers = defaultdict(float)

    for test_set in test_sets:

        train_matrix, train_values, test_matrix, test_values = get_test_train_set(data_matrix, actual_values, \
            test_set, is_distance_matrix)

        for model, model_name in zip(list_of_models, model_names):
            time_start = clock()
            predicted = get_predictions(model, train_matrix, train_values, test_matrix)
            timers[model_name] += clock() - time_start

            predictions[model_name].append((predicted, test_values))

    return predictions, timers

def plot_data(data_matrix, class_labels):
    """ For when you just want to look at some data """ 
    kpca = KernelPCA(kernel='cosine')
    pca = PCA()
    kpca_data_matrix = kpca.fit_transform(data_matrix)
    pca_data_matrix = pca.fit_transform(data_matrix)
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('PCA Space')

    unique_labels = set(class_labels)
    label_selectors = [ class_labels == label for label in unique_labels ] 

    for label, select in zip(unique_labels, label_selectors):
        plt.plot(pca_data_matrix[select, 0], pca_data_matrix[select, 1], 'o', label=label) 
    plt.legend() 

    plt.subplot(1, 2, 2)
    plt.title('Kernel PCA Space')
    for label, select in zip(unique_labels, label_selectors):
        plt.plot(kpca_data_matrix[select, 0], kpca_data_matrix[select, 1], 'o', label=label) 
    plt.show()

def build_model(sk_module, sk_model, **kwargs):
    """ Eventually, this function will build a classifer from a list of 
        valid models... or be deleted.  Likely the latter """ 
    model = getattr(getattr(__import__('sklearn', fromlist=[sk_module]), sk_module), sk_model)()
    model.set_params(**kwargs)
    return model
          
def build_list_of_classifiers(sklearn_file=None):
    """ Build a list of classifiers based on either a scikit-learn configuration file, or 
        using default values. 
    """ 
    if sklearn_file == None:
        classifiers = [ build_model('svm','SVC', **{'kernel':'rbf'}), \
                build_model('svm','SVC', **{'kernel':'linear'}), \
                build_model('ensemble', 'RandomForestClassifier', **{'n_estimators': 10}) ]
        classifier_names = ['svm-rbf', 'svm-lin', 'rf'] 
    else:
        classifiers, classifier_names = build_models_from_sklearn_file(sklearn_file)
    
    return classifiers, classifier_names

def build_list_of_regressors(sklearn_file=None):
    """ Build list of regression models, either from a default list or 
        one supplied in an sklearn configuration file.
    """
    if sklearn_file == None:
        models = [ build_model('svm','SVR', **{'kernel':'rbf'}), \
                build_model('svm','SVR', **{'kernel':'linear'}), \
                build_model('ensemble', 'RandomForestRegressor', **{'n_estimators': 10}) ]
        model_names = ['svm-rbf', 'svm-lin', 'rf'] 
    else:
        classifiers, classifier_names = build_models_from_sklearn_file(sklearn_file)
    
    return classifiers, classifier_names

def build_models_from_sklearn_file(sklearn_file):
    """ Build a list of models from configuration file """ 
    models = []
    model_names = []
    with open(sklearn_file, 'r') as f:
        for line in f:
            if line[0] != '#':
                input = line.strip().split('\t')
                name = input[1]
                parameters = {}
                for p in input[2:]:
                    param = p.split('=')
                    if param[0] == 'name':
                        name = param[1]
                    else:
                        parameters[param[0]] = custom_cast(param[1])
                model = build_model(input[0], input[1], **parameters)
                models.append(model)
                model_names.append(name)
    return models, model_names


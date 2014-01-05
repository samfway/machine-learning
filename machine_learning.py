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

from numpy import asarray, array
from random import shuffle
from biom.parse import parse_biom_table
from biom.table import DenseTable
from qiime.parse import parse_mapping_file_to_dict
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.metrics import classification_report, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
import warnings

def parse_otu_matrix(biom_file):
    """ Parses a (dense) OTU matrix from a biom file. 
        Outputs: Dense OTU matrix, list of sample ids
    """
    # Parse the OTU table into a dense matrix
    otu_table = parse_biom_table(open(biom_file,'U'))
    if isinstance(otu_table, DenseTable):
        otu_matrix = otu_table._data.T
    else:
        otu_matrix = asarray([v for v in otu_table.iterSampleData()])
    return otu_matrix, array(otu_table.SampleIds)

def parse_mapping_file_to_labels(mapping_file, sample_ids, metadata_category, metadata_value=None):
    """ Extracts the specified metadata category from the mapping file for each of the 
        sample ids in sample_ids.  Returns a boolean list of values if metadata_value is supplied """
    mapping_fp = open(mapping_file, 'rU')
    mapping_dict, comments = parse_mapping_file_to_dict(mapping_fp)

    class_labels = [ mapping_dict[sample_id][metadata_category] for sample_id in sample_ids]
    if metadata_value is not None:
        class_labels = [ label==metadata_value for label in class_labels ]
        if True not in class_labels:
            raise ValueError('No samples have the specified metadata_value (%s)' % \
                (metadata_value))
    return array(class_labels)

def parse_labels_file_to_dict(labels_file):
    """ Parse id-label file to a dictionary """ 
    label_dict = {}
    for line in open(labels_file, 'rU'):
        pieces = line.strip().split('\t')
        if len(pieces) != 2: continue 
        label_dict[pieces[0]] = pieces[1]
    return label_dict

def sync_labels_and_otu_matrix(otu_matrix, sample_ids, labels_dict):
    """ Returns appropriate rows of an otu matrix and corresponding sample id vector 
        to include all sample ids in labels_dict """ 
    select = array([ i for i in xrange(len(otu_matrix)) if sample_ids[i] in labels_dict.keys()])
    otu_matrix = otu_matrix[select, :]
    sample_ids = sample_ids[select]
    class_labels = array([labels_dict[sid] for sid in sample_ids])
    return otu_matrix, sample_ids, class_labels

def reduce_dimensionality(otu_matrix, k=10): 
    """ Performs truncated SVD on OTU matrix """ 
    tsvd = TruncatedSVD(k) 
    return tsvd.fit_transform(otu_matrix)

def get_test_sets(class_labels, kfold=10):
    """ Generate lists of indices for performing k-fold cross validation
        Note: test sets are stratefied, meaning that the number of class labels
        in each test set is proportional to the number in the whole set. 
    """
    return StratifiedKFold(class_labels, kfold)

def compare_tSVD(classifier, otu_matrix, class_labels, sample_ids, test_sets):
    """ Compare performance with/without truncated SVD """ 
    svd_otu_matrix = reduce_dimensionality(otu_matrix)
    print 'WITHOUT tSVD:'
    evaluate_classifier(classifier, otu_matrix, class_labels, sample_ids, test_sets)
    print '\nWITH tSVD:'
    evaluate_classifier(classifier, svd_otu_matrix, class_labels, sample_ids, test_sets)

def compare_classifiers(list_of_classifiers, classifier_names, otu_matrix, class_labels, sample_ids, test_sets):
    """ Evaluates each classifier in a list of classifiers, for a given test set """ 
    unique_labels = list(set(class_labels))
    label_length = max([len(label) for label in unique_labels])
    label_dict = { key:value for key, value in zip(unique_labels, range(len(unique_labels))) } 
    perf_measures = [ tuple(evaluate_classifier(classifier, otu_matrix, class_labels, sample_ids, test_sets)) \
        for classifier in list_of_classifiers ]
    for j in xrange(len(unique_labels)):
        print unique_labels[j]
        for i in xrange(len(list_of_classifiers)):
            precision = perf_measures[i][0][:, j]
            recall = perf_measures[i][1][:, j]
            f1 = array([ (2*p*r)/(p+r) for p,r in zip(precision,recall) ])
            p_dev = precision.std()*2
            r_dev = recall.std()*2
            f_dev = f1.std()*2
            p_mean = precision.mean() 
            r_mean = recall.mean()
            f_mean = f1.mean()
            name = classifier_names[i]
            print '[%s]%s\tPrecision: %.2f (+/- %.2f)\tRecall: %.2f (+/- %.2f) \t F1: %.2f (+/- %.2f)' % \
                (name, ' '*(label_length-len(name)), p_mean, p_dev, r_mean, r_dev, f_mean, f_dev)   
        print '' 

def evaluate_classifier(classifier, otu_matrix, class_labels, sample_ids, test_sets):
    """ Returns precision and recall measures for the provided classifier on each
        of the test sets given.  
    """
    unique_labels = list(set(class_labels))
    label_dict = { key:value for key, value in zip(unique_labels, range(len(unique_labels))) } 

    precision_scores = [] 
    recall_scores = [] 

    for train, test in test_sets:
        dev_data = otu_matrix[train,:]
        dev_labels = class_labels[train]
        dev_ids = sample_ids[train]  
        test_data = otu_matrix[test,:]
        test_labels = class_labels[test]
        test_ids = sample_ids[test]

        classifier.fit(dev_data, dev_labels)
        predictions = classifier.predict(test_data) 
        #print(classification_report(test_labels, predictions))
        # classification report, you snazzy... 

        test_labels_int = [ label_dict[x] for x in test_labels ] 
        predictions_int = [ label_dict[x] for x in predictions ] 
        
        """ If no predictions are made for class c, scikit raises a warning about there
            being no true or false positives.  With such a small amount of data, this is 
            entirely possible.  What's not possible is for there to be no true positives 
            or false negatives.  This is because we're using stratefied sampling to ensure
            that CV folds have similar proportions to the entire dataset. 

            Relevant discussion on sourceforge: 
            https://www.mail-archive.com/scikit-learn-general@lists.sourceforge.net/msg08862.html
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision_scores.append(precision_score(test_labels_int, predictions_int, average=None))
            recall_scores.append(recall_score(test_labels_int, predictions_int, average=None))

    precision_scores = array(precision_scores)
    recall_scores = array(recall_scores)
    return precision_scores, recall_scores

def plot_data(otu_matrix, class_labels, sample_ids):
    """ For when you just want to look at some damn data """ 
    kpca = KernelPCA(kernel='cosine')
    pca = PCA()
    kpca_otu_matrix = kpca.fit_transform(otu_matrix)
    pca_otu_matrix = pca.fit_transform(otu_matrix)
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('PCA Space')

    unique_labels = set(class_labels)
    label_selectors = [ class_labels == label for label in unique_labels ] 

    for label, select in zip(unique_labels, label_selectors):
        plt.plot(pca_otu_matrix[select, 0], pca_otu_matrix[select, 1], 'o', label=label) 
    plt.legend() 

    plt.subplot(1, 2, 2)
    plt.title('Kernel PCA Space')
    for label, select in zip(unique_labels, label_selectors):
        plt.plot(kpca_otu_matrix[select, 0], kpca_otu_matrix[select, 1], 'o', label=label) 
    plt.show()

def build_classifier(*args, **kwargs):
    """ Eventually, this function will build a classifer from a list of 
        valid classifiers... or be deleted.  Likely the latter """ 
    return build_rf()

def build_svm():
    """ Builds a standard SVM classifier """ 
    classifier = svm.SVC()
    return classifier 

def build_rf():
    """ Builds a standard Random Forest classifier """ 
    classifier = RandomForestClassifier(n_estimators=10)
    return classifier 

def build_list_of_classifiers():
    """ Get list of different classifiers """ 
    classifiers = [ svm.SVC(), \
        RandomForestClassifier(n_estimators=10), \
        NearestCentroid(), \
        AdaBoostClassifier() ] 
    classifier_names = ['svm', 'rf', 'nsc', 'adaboost']
    return classifiers, classifier_names
    
        

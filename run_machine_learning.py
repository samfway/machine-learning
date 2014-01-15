#!/usr/bin/env python
from numpy import array
import argparse
import machine_learning as ML
import sys

def interface():
    args = argparse.ArgumentParser(
        description='Simple wrapper for machine_learning.py.',
        epilog="""Supply the following: 
                  * OTU table (.biom) 
                  And one of the following:
                  * Mapping file + Metadata category + (optional) Metadata value
                  * Separate label file 
                """)
    args.add_argument('-i', '--biom-file', help='Biom OTU table', required=True)
    args.add_argument('-m', '--mapping-file', help='Mapping table')
    args.add_argument('-l', '--labels-file', help='Labels file')
    args.add_argument('-n', '--num-folds', help='Number of folds for cross-val', \
                            default=10, type=int)
    args.add_argument('-c', '--metadata-category', help='Metadata category')
    args.add_argument('-v', '--metadata-value', help='Metadata value') 
    args.add_argument('-s', '--sklearn-file', help='Scikit-learn configuration file')
    args.add_argument('-f', '--find-features', help='Find best features for each classifier', \
                            action='store_true')
    args.add_argument('-o', '--output-file', help='Output file for report (default: ML-report.txt)', \
                            default='ML-report.txt', type=str)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    
    if args.mapping_file == None and args.labels_file == None:
        print 'You must supply either a mapping file + metadata category OR a labels file'
        exit() 

    otu_matrix, sample_ids = ML.parse_otu_matrix(args.biom_file)

    if args.mapping_file is not None:
        if args.metadata_category is None:
            print "To extract labels from a mapping file, you must supply the desired " + \
                  "metadata category!"
            exit()
        class_labels = ML.parse_mapping_file_to_labels(args.mapping_file, sample_ids, \
            args.metadata_category, args.metadata_value)
    else:
        sample_ids = array([x.split('.')[0] for x in sample_ids]) # Hack to work with Dan's stuff
        label_dict =  ML.parse_labels_file_to_dict(args.labels_file)
        otu_matrix, sample_ids, class_labels = ML.sync_labels_and_otu_matrix(otu_matrix, \
            sample_ids, label_dict)

    list_of_classifiers, name_of_classifiers = ML.build_list_of_classifiers(args.sklearn_file)
    test_sets = ML.get_test_sets(class_labels, args.num_folds)

    ML.compare_classifiers(list_of_classifiers, name_of_classifiers, otu_matrix, class_labels, \
        sample_ids, test_sets, args.find_features, args.output_file)

    #otu_matrix = ML.reduce_dimensionality(otu_matrix)

    #print '--- AFTER REDUCING DIMENSIONALITY TO k=10 ---' 
    #ML.compare_classifiers(list_of_classifiers, name_of_classifiers, otu_matrix, class_labels, \
    #    sample_ids, test_sets)

    #ML.plot_data(otu_matrix, class_labels, sample_ids)
    #classifier = ML.build_classifier()
    #training_sets = ML.get_test_sets(class_labels, args.num_folds)
    #ML.compare_tSVD(classifier, otu_matrix, class_labels, sample_ids, training_sets)
    #ML.evaluate_classifier(classifier, otu_matrix, class_labels, sample_ids, training_sets)



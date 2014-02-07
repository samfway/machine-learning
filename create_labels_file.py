#!/usr/bin/env python
from numpy import array
import argparse
from argparse import RawTextHelpFormatter
import ml_lib.parse as ml_parse

def interface():
    args = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description='Create labels file from mapping file.',
        epilog='Allows for easier, external preprocessing of labels.')
    args.add_argument('-m', '--mapping-file', help='Mapping table', required=True)
    args.add_argument('-c', '--metadata-category', help='Metadata category', required=True)
    args.add_argument('-o', '--output-file', help='Output labels file (default: labels.txt)', \
                            default='labels.txt', type=str)
    args = args.parse_args()
    return args

def create_labels_file(mapping_file, metadata_category, labels_file, simple_id=True):
    """ Extract metadata category from mapping file into separate labels file 
        simple_id specifies whether or not to simplify sample id's ala Dan Knights.
        For example "M24Plml.140651", becomes "M24Plml"
    """
    label_dict = ml_parse.parse_metadata_category_from_mapping_file(mapping_file, \
        metadata_category)
    output = open(labels_file, 'w')
    output.write('label\n')
    for key, value in label_dict.iteritems():   
        if simple_id: key = key.split('.')[0] 
        output.write('%s\t%s\n' % (key, str(value)))
    output.close() 

if __name__=="__main__":
    args = interface()
    create_labels_file(args.mapping_file, args.metadata_category, args.output_file)


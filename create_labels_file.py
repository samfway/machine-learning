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
    args.add_argument('-c', '--metadata-category', help='Metadata category')
    args.add_argument('-v', '--metadata-value', help='Specific metadata value')
    args.add_argument('-o', '--output-file', help='Output labels file (default: labels.txt)', \
                            default='labels.txt', type=str)
    args.add_argument('--view', help='View metadata categories', action='store_true', default=False)
    args = args.parse_args()
    return args

def create_labels_file(mapping_file, metadata_category, labels_file, simple_id=True, metadata_value=None):
    """ Extract metadata category from mapping file into separate labels file 
        simple_id specifies whether or not to simplify sample ids ala Dan Knights.
        For example "M24Plml.140651", becomes "M24Plml"
    """
    label_dict = ml_parse.parse_metadata_category_from_mapping_file(mapping_file, \
        metadata_category)
    output = open(labels_file, 'w')
    output.write('label\n')
    for key, value in label_dict.iteritems():   
        if simple_id: key = key.split('.')[0] 
        if metadata_value is not None: value = str(value) in metadata_value
        output.write('%s\t%s\n' % (key, str(value)))
    output.close() 

if __name__=="__main__":
    args = interface()
    if args.view:
        ml_parse.view_metadata_categories_from_mapping_file(args.mapping_file)
    else:
        if args.metadata_category is None:
            print 'You must supply a metadata category to extract.  Exiting'
            exit() 
        create_labels_file(args.mapping_file, args.metadata_category, args.output_file, \
            simple_id=True, metadata_value=args.metadata_value)


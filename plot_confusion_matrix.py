#!/usr/bin/env python
import argparse
from ml_lib.plot_confusion_matrix import plot_confusion_matrix
from ml_lib.parse import parse_confusion_matrix_file

def interface():
    args = argparse.ArgumentParser(
        description='Simple wrapper for plot_confusion_matrix.py.',
        epilog="""Create a confusion matrix plot for the supplied confusion matrix file
                """)
    args.add_argument('-i', '--input-file', help='Confusion matrix file', required=True)
    args.add_argument('-o', '--output-file', help='Output filename', default='confusion_matrix.pdf')
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    confusion_matrix, labels = parse_confusion_matrix_file(args.input_file, normalized=True)
    plot_confusion_matrix(confusion_matrix, labels, args.output_file)
    

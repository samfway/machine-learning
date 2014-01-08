#!/usr/bin/env python
import argparse
import plot_confusion_matrix as pcm 

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
    confusion_matrix, labels = pcm.parse_confusion_matrix_file(args.input_file, normalized=True)
    pcm.make_confusion_matrix_plot(confusion_matrix, labels, args.output_file)
    

import argparse
import os

from segmentation_utils import filter_and_clean_directory

arg_parser = argparse.ArgumentParser(description="preprocessor for workman images")
arg_parser.add_argument('--input_dir', help='input dir to clean images from')
arg_parser.add_argument('--output_dir', help='output dir to mv images from')
args = arg_parser.parse_args()

def preprocess_images(input_dir, output_dir):
    # filter_directory removes images in place so 
    filter_and_clean_directory(input_dir, output_dir)
    
if __name__ == '__main__':
    preprocess_images(**vars(args))

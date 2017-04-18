#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" problem1.py: This module implements a Perceptron (edx.org - ColumbiaX - AI Week7)"""

__author__ = "Selam Getachew Woldetsadick"
__copyright__ = "Copyright (c) 2017 Selam Getachew Woldetsadick"


def csv_dataset_reader(path):
    """
    This function reads a csv from a specified path and returns a Pandas dataframe representation of it, and renames
    columns.
    :param path: Path to and name of the csv file to read.
    :return: A Pandas dataframe.
    """
    import pandas as pd
    data = pd.read_csv(path, sep=",", header=None)
    data.columns = ['feature_1', 'feature_2', 'label']
    return data


def parse_system_arguments():
    """
    This function parses system arguments passed with python script launch.
    :return: tuple of input and output file names.
    """
    import sys
    input_t, output_t = sys.argv[1], sys.argv[2]
    return str(input_t), str(output_t)


if __name__ == "__main__":
    in_put, out_put = parse_system_arguments()
    data_frame = csv_dataset_reader("./" + in_put)
    print data_frame.shape

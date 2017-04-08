#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" problem2.py: This module implements a Linear Regression example(edx.org - ColumbiaX - AI Week 7)"""

__author__ = "Selam Getachew Woldetsadick"
__copyright__ = "Copyright (c) 2017 Selam Getachew Woldetsadick"


def parse_system_arguments():
    """
    This function parses system arguments passed with python script launch.
    :return: tuple of input and output file names.
    """
    import sys
    input, output = sys.argv[1], sys.argv[2]
    return str(input), str(output)


def csv_dataset_reader(path):
    """
    This function reads a csv from a specified path and returns a Pandas dataframe representation of it, and renames
    columns.
    :param path: Path to and name of the csv file to read.
    :return: A Pandas dataframe.
    """
    import pandas as pd
    data = pd.read_csv(path, sep=",", header=None)
    data.columns = ['age', 'weight', 'height']
    return data


def center_and_scale_data(data):
    """
     This function reads a Pandas dataframe and centers and scales the columns and add an intercept vector.
     :param data: A pandas dataframe.
     :return: A Pandas dataframe with scaled and centered data.
     """
    a_mean, a_std = datum['age'].mean(), datum['age'].std()
    datum['age'] = datum['age'].apply(lambda x: ((x - a_mean) / a_std))
    w_mean, w_std = datum['weight'].mean(), datum['weight'].std()
    datum['weight'] = datum['weight'].apply(lambda x: ((x - w_mean) / w_std))
    datum['intercept'] = datum['height'].map(lambda x: 1 if x else 1)
    return datum


if __name__ == "__main__":
    input_file, output_file = parse_system_arguments()
    datum = csv_dataset_reader("./" + input_file)
    standardized_datum = center_and_scale_data(datum)

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
    input_t, output_t = sys.argv[1], sys.argv[2]
    return str(input_t), str(output_t)


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
    a_mean, a_std = data['age'].mean(), data['age'].std()
    data['age'] = data['age'].apply(lambda x: ((x - a_mean) / a_std))
    w_mean, w_std = data['weight'].mean(), data['weight'].std()
    data['weight'] = data['weight'].apply(lambda x: ((x - w_mean) / w_std))
    data['intercept'] = data['height'].map(lambda x: 1 if x else 1)
    return data


def output_csv_writer(output_path, method, best_score, test_score):
    """
    This function writes an output in a file called output3.csv
    :param output_path: The name and path in/with which the output file is written.
    :param method: The method used to build classification
    :param best_score: Best accuracy score on parameters grid in 5-folds CV (float)
    :param test_score: Accuracy score on test set
    :return: None
    """
    with open("./" + output_path, 'w') as f:
        f.write("%s,%f,%f\n" % (str(method), float(best_score), float(test_score)))
    f.close()
    return


if __name__ == "__main__":
    input_file, output_file = parse_system_arguments()
    datum = csv_dataset_reader("./" + input_file)
    standardized_datum = center_and_scale_data(datum)

    # output_csv_writer(output_file, method, best_score, test_score)

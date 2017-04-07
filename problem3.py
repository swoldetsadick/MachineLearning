#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""problem3.py: This module implements a small Classification in sklearn (edx.org - ColumbiaX - AI Week 7)"""

__author__ = "Selam Getachew Woldetsadick"
__copyright__ = "Copyright (c) 2017 Selam Getachew Woldetsadick"


def features_labels_extractor(data):
    """
    This function takes a Pandas representation of data, and transforms it to Numpy representation, while splitting data
    into features and labels arrays
    :param data: A Pandas dataframe
    :return: A tuple of two elements respectively features and labels arrays
    """
    features = data.as_matrix(columns=['A', 'B'])
    labels = data.as_matrix(columns=['label'])
    return features, labels


def csv_dataset_reader(path="./input3.csv"):
    """
    This function reads a csv from a specified path and returns a Pandas dataframe representation of it.
    :param path: Path to and name of the csv file to read (default:"./input3.csv")
    :return: A Pandas dataframe
    """
    import pandas as pd
    data = pd.read_csv(path, sep=",", header="infer")
    return data


def train_test_splitter(features, labels):
    """
    This function splits data into training (60%) and testing (40%), making sure of stratified sampling
    :param features: A set of features in Numpy array format
    :param labels: A set of labels in Numpy array format
    :return: a four elements tuples respectively training features, testing features, training labels and testing labels
    """
    from sklearn.model_selection import train_test_split
    f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=0.4, random_state=10,
                                                        stratify=labels)
    return f_train, f_test, l_train, l_test


if __name__ == "__main__":
    datum = csv_dataset_reader()
    feats, lab = features_labels_extractor(datum)
    feats_train, lab_test, feats_train, lab_test = train_test_splitter(feats, lab)

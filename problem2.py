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


def output_csv_writer(output_path, learning_rate, converge, beta_0, beta_1, beta_2):
    """
    This function writes an output in a file whose name is given by first parameter.
    :param output_path: The name and path in/with which the output file is written.
    :param learning_rate: The learning rate used.
    :param converge: an int representing the number of iteration of GD algorithm.
    :param beta_0: weight associated to the intercept.
    :param beta_1: weight associated to the first feature "age".
    :param beta_2: weight associated to the second feature "weight".
    :return: None
    """
    with open("./" + output_path, 'a') as f:
        f.write("%s,%s,%s,%s,%s\n" % (learning_rate, converge, beta_0, beta_1, beta_2))
    f.close()
    return


def loss_function(data, beta_0=0, beta_1=0, beta_2=0):
    """
    This function calculates the squared error between estimations and real observations
    :param data: Pandas dataframe on which estimation is conducted
    :param beta_0: weight associated to the intercept.
    :param beta_1: weight associated to the first feature "age".
    :param beta_2: weight associated to the second feature "weight".
    :return: Mean Squared Error of Estimation. 
    """
    data['error'] = data.apply(lambda row: square_of_error(row['height'], beta_0, row['intercept'], beta_1, row['age'],
                                                           beta_2, row['weight']), axis=1)
    error = data['error'].sum()
    data.drop('error', axis=1, inplace=True)
    return error


def square_of_error(y, beta_0, x_0, beta_1, x_1, beta_2, x_2, principal=None, gradient=False, data_length=0):
    """
    This function calculates the squared errors.
    :param y: label of the dataframe height value
    :param beta_0: weight associated to the intercept
    :param x_0: intercept value
    :param beta_1: weight associated to the first feature "age"
    :param x_1: age value
    :param beta_2: weight associated to the second feature "weight"
    :param x_2: weight value
    :param principal: optional if for gradient descent indicates feature considered
    :param gradient: a boolean if estimation is used for gradient or not
    :param data_length: length of estimated data.
    :return: Squared Error if not gradient, gradient step if gradient
    """
    if not gradient:
        return (y - (beta_0 * x_0 + beta_1 * x_1 + beta_2 * x_2)) ** 2
    else:
        return -(1 / float(data_length)) * principal * ((beta_0 * x_0 + beta_1 * x_1 + beta_2 * x_2) - y)


def one_gradient_step(data, alpha, beta_0=0, beta_1=0, beta_2=0):
    """
    This function calculates a gradient step.
    :param data: Pandas dataframe on which estimation is conducted
    :param alpha: a float for learning rate
    :param beta_0: weight associated to the intercept
    :param beta_1: weight associated to the first feature "age"
    :param beta_2: weight associated to the second feature "weight"
    :return: Mean Squared Error of Estimation. 
    """
    l = data.shape[0]
    data['err_it'] = data.apply(
        lambda row: square_of_error(row['height'], beta_0, row['intercept'], beta_1, row['age'], beta_2, row['weight'],
                                    principal=row['intercept'], gradient=True, data_length=l), axis=1)
    data['err_ag'] = data.apply(
        lambda row: square_of_error(row['height'], beta_0, row['intercept'], beta_1, row['age'], beta_2, row['weight'],
                                    principal=row['age'], gradient=True, data_length=l), axis=1)
    data['err_wt'] = data.apply(
        lambda row: square_of_error(row['height'], beta_0, row['intercept'], beta_1, row['age'], beta_2, row['weight'],
                                    principal=row['weight'], gradient=True, data_length=l), axis=1)
    beta_0 += alpha * data['err_it'].sum()
    beta_1 += alpha * data['err_ag'].sum()
    beta_2 += alpha * data['err_wt'].sum()
    data.drop('err_it', axis=1, inplace=True)
    data.drop('err_ag', axis=1, inplace=True)
    data.drop('err_wt', axis=1, inplace=True)
    return beta_0, beta_1, beta_2


def converge_it(data, converge_num, alphas, out_path):
    losses = []
    while alphas:
        alpha = alphas.pop(0)
        b_0, b_1, b_2 = 0, 0, 0
        i = 0
        if not alphas:
            converge_num = 500
        while i < converge_num:
            b_0, b_1, b_2 = one_gradient_step(data, alpha, b_0, b_1, b_2)
            i += 1
        losses.append(loss_function(data, b_0, b_1, b_2))
        output_csv_writer(out_path, alpha, converge_num, b_0, b_1, b_2)
    return losses


if __name__ == "__main__":
    input_file, output_file = parse_system_arguments()
    datum = csv_dataset_reader("./" + input_file)
    standardized_datum = center_and_scale_data(datum)
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, .7]
    iteration = 100
    loser = converge_it(standardized_datum, iteration, learning_rates, output_file)

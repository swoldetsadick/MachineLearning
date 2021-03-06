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
    labels = labels.flatten()
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


def graph_3d_my_data(data):
    """
    This function 3D graphs my data.
    :param data: Graphed data with three columns
    :return: None
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = data.shape[0]
    for i in range(0, n):
        xs = data['A'][i]
        ys = data['B'][i]
        zs = data['label'][i]
        if zs == 0:
            a = 'r'
            b = 'o'
        else:
            a = 'b'
            b = '^'
        ax.scatter(xs, ys, zs, c=a, marker=b)
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('Labels')
    plt.show()
    return


def linear_svm_classification(training_set_features, testing_set_features, training_set_labels, testing_set_labels):
    """
    This function conducts a linear kernel SVM classification with 5 folds CV using a grid search to fit to best 
    learning rate
    :param training_set_features: multi-dimensional array representing training set features.
    :param testing_set_features: multi-dimensional array representing testing set features.
    :param training_set_labels: uni-dimensional array representing training set labels.
    :param testing_set_labels: uni-dimensional array representing testing set labels.
    :return: Three elements tuple respectively method used (String), best accuracy score on parameters grid in 5-folds 
    CV (float), accuracy score on test set
    """
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    method = "svm_linear"
    scaler = StandardScaler()
    scaled_feats_train = scaler.fit_transform(training_set_features)
    svr = svm.SVC(kernel='linear', random_state=10)
    parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
    clf = GridSearchCV(svr, parameters, cv=5, scoring='accuracy')
    clf.fit(scaled_feats_train, training_set_labels)
    scaled_feats_test = scaler.transform(testing_set_features)
    predicted_lab_test = clf.predict(scaled_feats_test)
    best_score = clf.best_score_
    test_score = accuracy_score(testing_set_labels, predicted_lab_test, normalize=True)
    return method, best_score, test_score


def polynomial_svm_classification(training_set_features, testing_set_features, training_set_labels, testing_set_labels):
    """
    This function conducts a polynomial kernel SVM classification with 5 folds CV using a grid search to fit to best 
    learning rate, degree of polynomial and the kernel coefficient gamma
    :param training_set_features: multi-dimensional array representing training set features.
    :param testing_set_features: multi-dimensional array representing testing set features.
    :param training_set_labels: uni-dimensional array representing training set labels.
    :param testing_set_labels: uni-dimensional array representing testing set labels.
    :return: Three elements tuple respectively method used (String), best accuracy score on parameters grid in 5-folds 
    CV (float), accuracy score on test set
    """
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    method = "svm_polynomial"
    scaler = StandardScaler()
    scaled_feats_train = scaler.fit_transform(training_set_features)
    svr = svm.SVC(kernel='poly', random_state=10)
    parameters = {'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 1]}
    clf = GridSearchCV(svr, parameters, cv=5, scoring='accuracy')
    clf.fit(scaled_feats_train, training_set_labels)
    scaled_feats_test = scaler.transform(testing_set_features)
    predicted_lab_test = clf.predict(scaled_feats_test)
    best_score = clf.best_score_
    test_score = accuracy_score(testing_set_labels, predicted_lab_test, normalize=True)
    return method, best_score, test_score


def rbf_svm_classification(training_set_features, testing_set_features, training_set_labels, testing_set_labels):
    """
    This function conducts a rbf kernel SVM classification with 5 folds CV using a grid search to fit to best 
    learning rate and the kernel coefficient gamma
    :param training_set_features: multi-dimensional array representing training set features.
    :param testing_set_features: multi-dimensional array representing testing set features.
    :param training_set_labels: uni-dimensional array representing training set labels.
    :param testing_set_labels: uni-dimensional array representing testing set labels.
    :return: Three elements tuple respectively method used (String), best accuracy score on parameters grid in 5-folds 
    CV (float), accuracy score on test set
    """
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    method = "svm_rbf"
    scaler = StandardScaler()
    scaled_feats_train = scaler.fit_transform(training_set_features)
    svr = svm.SVC(kernel='rbf', random_state=10)
    parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}
    clf = GridSearchCV(svr, parameters, cv=5, scoring='accuracy')
    clf.fit(scaled_feats_train, training_set_labels)
    scaled_feats_test = scaler.transform(testing_set_features)
    predicted_lab_test = clf.predict(scaled_feats_test)
    best_score = clf.best_score_
    test_score = accuracy_score(testing_set_labels, predicted_lab_test, normalize=True)
    return method, best_score, test_score


def logistic_classification(training_set_features, testing_set_features, training_set_labels, testing_set_labels):
    """
    This function conducts a logistic regression with 5 folds CV using a grid search to fit to best learning rate
    :param training_set_features: multi-dimensional array representing training set features.
    :param testing_set_features: multi-dimensional array representing testing set features.
    :param training_set_labels: uni-dimensional array representing training set labels.
    :param testing_set_labels: uni-dimensional array representing testing set labels.
    :return: Three elements tuple respectively method used (String), best accuracy score on parameters grid in 5-folds 
    CV (float), accuracy score on test set
    """
    from sklearn import linear_model
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    method = "logistic"
    svr = linear_model.LogisticRegression(random_state=10)
    parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
    clf = GridSearchCV(svr, parameters, cv=5, scoring='accuracy')
    clf.fit(training_set_features, training_set_labels)
    predicted_lab_test = clf.predict(testing_set_features)
    best_score = clf.best_score_
    test_score = accuracy_score(testing_set_labels, predicted_lab_test, normalize=True)
    return method, best_score, test_score


def knn_classification(training_set_features, testing_set_features, training_set_labels, testing_set_labels):
    """
    This function conducts a K nearest neighbours classification with 5 folds CV using a grid search to fit to best 
    number of number and number of leafs
    :param training_set_features: multi-dimensional array representing training set features.
    :param testing_set_features: multi-dimensional array representing testing set features.
    :param training_set_labels: uni-dimensional array representing training set labels.
    :param testing_set_labels: uni-dimensional array representing testing set labels.
    :return: Three elements tuple respectively method used (String), best accuracy score on parameters grid in 5-folds 
    CV (float), accuracy score on test set
    """
    from sklearn import neighbors
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    method = "knn"
    scaler = StandardScaler()
    scaled_feats_train = scaler.fit_transform(training_set_features)
    svr = neighbors.KNeighborsClassifier(metric='euclidean')
    parameters = {'n_neighbors': range(1, 51), 'leaf_size': [i for i in range(1, 61) if i % 5 == 0]}
    clf = GridSearchCV(svr, parameters, cv=5, scoring='accuracy')
    clf.fit(scaled_feats_train, training_set_labels)
    scaled_feats_test = scaler.transform(testing_set_features)
    predicted_lab_test = clf.predict(scaled_feats_test)
    best_score = clf.best_score_
    test_score = accuracy_score(testing_set_labels, predicted_lab_test, normalize=True)
    return method, best_score, test_score


def tree_classification(training_set_features, testing_set_features, training_set_labels, testing_set_labels):
    """
    This function conducts a tree classification with 5 folds CV using a grid search to fit to best number of maximum 
    tree depth and minimum number after which a leaf cannot split further.
    :param training_set_features: multi-dimensional array representing training set features.
    :param testing_set_features: multi-dimensional array representing testing set features.
    :param training_set_labels: uni-dimensional array representing training set labels.
    :param testing_set_labels: uni-dimensional array representing testing set labels.
    :return: Three elements tuple respectively method used (String), best accuracy score on parameters grid in 5-folds 
    CV (float), accuracy score on test set
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    method = "decision_tree"
    scaler = StandardScaler()
    scaled_feats_train = scaler.fit_transform(training_set_features)
    svr = DecisionTreeClassifier(random_state=0)
    parameters = {'max_depth': range(1, 51), 'min_samples_split': range(2, 11)}
    clf = GridSearchCV(svr, parameters, cv=5, scoring='accuracy')
    clf.fit(scaled_feats_train, training_set_labels)
    scaled_feats_test = scaler.transform(testing_set_features)
    predicted_lab_test = clf.predict(scaled_feats_test)
    best_score = clf.best_score_
    test_score = accuracy_score(testing_set_labels, predicted_lab_test, normalize=True)
    return method, best_score, test_score


def rf_classification(training_set_features, testing_set_features, training_set_labels, testing_set_labels):
    """
    This function conducts a random forest classification with 5 folds CV using a grid search to fit to best number of 
    maximum tree depth and minimum number after which a leaf cannot split further.
    :param training_set_features: multi-dimensional array representing training set features.
    :param testing_set_features: multi-dimensional array representing testing set features.
    :param training_set_labels: uni-dimensional array representing training set labels.
    :param testing_set_labels: uni-dimensional array representing testing set labels.
    :return: Three elements tuple respectively method used (String), best accuracy score on parameters grid in 5-folds 
    CV (float), accuracy score on test set
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    method = "random_forest"
    scaler = StandardScaler()
    scaled_feats_train = scaler.fit_transform(training_set_features)
    svr = RandomForestClassifier(random_state=0)
    parameters = {'max_depth': range(1, 51), 'min_samples_split': range(2, 11)}
    clf = GridSearchCV(svr, parameters, cv=5, scoring='accuracy')
    clf.fit(scaled_feats_train, training_set_labels)
    scaled_feats_test = scaler.transform(testing_set_features)
    predicted_lab_test = clf.predict(scaled_feats_test)
    best_score = clf.best_score_
    test_score = accuracy_score(testing_set_labels, predicted_lab_test, normalize=True)
    return method, best_score, test_score


def output_csv_writer(method, best_score, test_score):
    """
    This function writes an output in a file called output3.csv
    :param method: The method used to build classification
    :param best_score: Best accuracy score on parameters grid in 5-folds CV (float)
    :param test_score: Accuracy score on test set
    :return: None
    """
    with open('./output3.csv', 'a') as f:
        f.write("%s,%f,%f\n" % (str(method), float(best_score), float(test_score)))
    f.close()
    return


if __name__ == "__main__":
    datum = csv_dataset_reader()
    # graph_3d_my_data(datum)
    feats, lab = features_labels_extractor(datum)
    feats_train, feats_test, lab_train, lab_test = train_test_splitter(feats, lab)
    m, b_score, t_score = linear_svm_classification(feats_train, feats_test, lab_train, lab_test)
    output_csv_writer(m, b_score, t_score)
    m, b_score, t_score = polynomial_svm_classification(feats_train, feats_test, lab_train, lab_test)
    output_csv_writer(m, b_score, t_score)
    m, b_score, t_score = rbf_svm_classification(feats_train, feats_test, lab_train, lab_test)
    output_csv_writer(m, b_score, t_score)
    m, b_score, t_score = logistic_classification(feats_train, feats_test, lab_train, lab_test)
    output_csv_writer(m, b_score, t_score)
    m, b_score, t_score = knn_classification(feats_train, feats_test, lab_train, lab_test)
    output_csv_writer(m, b_score, t_score)
    m, b_score, t_score = tree_classification(feats_train, feats_test, lab_train, lab_test)
    output_csv_writer(m, b_score, t_score)
    m, b_score, t_score = rf_classification(feats_train, feats_test, lab_train, lab_test)
    output_csv_writer(m, b_score, t_score)

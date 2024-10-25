import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from cleaning_data import *
from stats import *
from implementations import *
from utils import split_data, downsample_class_0, upsample_class_1_to_percentage
from functions import predict, slice_data
from cleaning_data import *
import datetime
import seaborn as sns
import os

def run():

    print("Beginning data loading")

    # Load data
    DATA_PATH = 'C:/Users/clavo\OneDrive - epfl.ch/EPFL/Cours/Semester 1/CS-433/Project/dataset/dataset' # Change to path for dataset.zip extracted folder
    x_train, x_test, y_train, train_ids, test_ids, labels =  load_csv_data(DATA_PATH, sub_sample=False)
    labels.pop(0)

    print("Loading complete")

    # NUMBER OF FOLDS IN K-FOLD CROSS VALIDATION
    k = 4

    # Split the data into k folds
    x_train_slices, y_train_slices = slice_data(x_train, y_train, k, seed=2)

    # Generating initial w array for each slice
    ws = np.zeros((k, x_train.shape[1]))

    for fold in range(k):

        print("Beginning processing on fold " + str(fold))

        # Getting test and train data for specific fold iteration
        x_te = x_train_slices[fold]
        y_te = y_train_slices[fold]

        x_tr = np.vstack(x_train_slices[:fold] + x_train_slices[fold+1:])
        y_tr = np.vstack(y_train_slices[:fold] + y_train_slices[fold+1:])

        # DATA PREPROCESSING HYPERPARAMETERS
        upsampling = 0.2                # Fraction of total data with positive labels (rebalancing dataset by randomly duplicating positively labelled datapoints)
        degree = 1                      # Polynomial expansion degree
        variance_threshold = 0.995      # Threshold for outlier removal
        nan_percentage = 1              # Threshold of NaN values in a feature above which the feature is removed
        outliers_row_lim = 0.2          # Threshold of values in a datapoint which are outliers (according to variance threshold) above which thed atapoint is removed

        print("Beginning data cleaning")

        # Clean training data
        x_train_cleaned, y_tr_cleaned, features, median_and_most_probable_class, W, mean = clean_train_data(x_tr, y_tr,labels, upsampling, degree, variance_threshold, nan_percentage, outliers_row_lim)

        # Clean testing data identically to training data cleaning specifications
        x_te_cleaned = clean_test_data(x_te, labels, features, median_and_most_probable_class, mean, W, degree)

        print("Cleaning complete")

        # Initializing initial guess for weights with Gaussian distribution
        mean = 0    # Mean of the distribution
        std_dev = 1 # Standard deviation of the distribution

        num_samples = x_train_cleaned.shape[0]
        tx = np.c_[np.ones(num_samples), x_train_cleaned]
        w_initial = np.random.normal(loc=mean, scale=std_dev, size=tx.shape[1])

        # ML MODEL HYPERPARAMETERS
        max_iter = 300
        gamma = 0.5
        lambda_ = 0.01
        threshold = 0.47    # Predicted percentage above which a positive label is assigned (default =0.5)

        print("Beginning training")

        # Training model with selected learning model
        ws[fold], loss = reg_logistic_regression(y_tr_cleaned, tx, lambda_, w_initial, max_iter, gamma)

        print("Training complete")

        # Predicting internal test labels and evaluating model accuracy
        num_samples = x_te_cleaned.shape[0]
        tx_te = np.c_[np.ones(num_samples), x_te_cleaned]
        y_predict = predict(tx_te, ws[fold], threshold)
        f1_score_te = f1_score(y_te, y_predict)

        print('Model prediction F1 score:')
        print(f1_score_te)

    w_mean = np.mean(ws, axis=0)

    # Predicting real data labels and exporting predictions in .csv

    print("Beginning official test data cleaning and label prediction")

    x_test_cleaned = clean_test_data(x_test, labels, features, median_and_most_probable_class, mean, W, degree)
    num_samples = x_te.shape[0]
    tx_te = np.c_[np.ones(num_samples), x_te_cleaned]

    num_samples = x_test_cleaned.shape[0]
    tx_test = np.c_[np.ones(num_samples), x_test_cleaned]
    y_test_to_save = predict(tx_test, w_mean, threshold)

    # Stack the ids and predictions together column-wise
    submit = np.column_stack((test_ids, y_test_to_save))

    print("Prediction complete, exporting results")

    # Save to a CSV file using np.savetxt
    EXPORTPATH = "C:/Users/clavo/OneDrive - epfl.ch/EPFL/Cours/Semester 1/CS-433/Project/ML_course/projects/project1/data"
    np.savetxt(EXPORTPATH, submit, delimiter=",", fmt='%d,%d', header='Id,Prediction', comments='')

    print("Export complete")

run()

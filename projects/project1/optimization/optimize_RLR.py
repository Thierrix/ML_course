import sys
import os


# Get the parent directory (one level above the current file)
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add the parent directory to sys.path
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from cleaning_data import *
from stats import *
from functions import *
from utils import split_data, downsample_class_0, upsample_class_1_to_percentage
from functions import *
from optimization.graphs import *

print("Beginning data loading")
DATA_PATH = '/Users/williamjallot/Desktop/ML/dataset'
x_train, x_test, y_train, train_ids, test_ids, labels =  load_csv_data(DATA_PATH, sub_sample=False)
labels.pop(0) 
print("Loading complete")
#Split the data into training and testing
x_tr, x_te, y_tr, y_te = split_data(x_train, y_train, 0.8, seed= 2)

# Define hyperparameters in a dictionary
hyperparameters = {
    "lambdas": [0.01],
    "up_sampling_percentages": [0.2, 0.25],
    "degrees": [1],
    "variances_threshold": [0.995],
    "decision_threshold": [0.46],
    "acceptable_nan_percentages": [1],
    "max_iters": [300],
    "outliers_row_limit": [0.7],
    "gammas": [0.5],
    "nan_handlers": ['numeric']
}

k_fold = 4

# MODEL SELECTION
models = ["reg_logistic_regression"]

best_params = grid_search_k_fold_logistic(models, y_tr, x_tr, k_fold, hyperparameters, labels)

# Unpack the best parameters
best_gamma = best_params['gamma']
best_up_sampling_percentage = best_params['up_sampling_percentage']
best_degree = best_params['degree']
best_variance_threshold = best_params['variance_threshold']
best_lambda = best_params['lambda_']
best_max_iter = best_params['max_iter']
best_threshold = best_params['threshold']
best_nan_percentage = best_params['acceptable_nan_percentage']
best_outlier_limit = best_params['outlier_limit']
best_nan_handler = best_params['nan_handling']

print(f'The best parameters are gamma = {best_gamma}, best_nan_handling = {best_nan_handler},up_sampling_percentage = {best_up_sampling_percentage}, degree = {best_degree}, variance_treshold = {best_variance_threshold}, lambda = {best_lambda}, threshold = {best_threshold},max_iter = {best_max_iter}, best nan percentage = {best_nan_percentage}')
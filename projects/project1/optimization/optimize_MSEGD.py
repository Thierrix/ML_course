import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from cleaning_data import *
from stats import *
from functions import *
from utils import split_data, downsample_class_0, upsample_class_1_to_percentage
from functions import *
import datetime
import seaborn as sns

DATA_PATH = '/Users/williamjallot/Desktop/ML/dataset'
x_train, x_test, y_train, train_ids, test_ids, labels =  load_csv_data(DATA_PATH, sub_sample=False)
labels.pop(0)

#Split the data into training and testing
x_tr, x_te, y_tr, y_te = split_data(x_train, y_train, 0.8, seed= 2)

lambdas = [0.01, 0.02]
up_sampling_percentages = [0.2, 0.22]
degrees = [1]
variances_threshold = [0.995]
decision_threshold = [0.47]
acceptable_nan_percentages = [1]
max_iters = [300]
outliers_row_limit = [0.2, 0.1]
gammas = [0.45]
nan_handlers = ['numeric', 'median', 'mean']
k_fold = 4

best_gamma, best_up_sampling_percentage, best_degree, best_variance_threshold, best_lambda,best_max_iter, best_f1_score, best_threshold, best_nan_percentage, best_outlier_limit, best_nan_handler = grid_search_k_fold(y_tr, x_tr, k_fold, lambdas, gammas, up_sampling_percentages,
                                                                                                                                                                                     degrees, variances_threshold, max_iters,decision_threshold, acceptable_nan_percentages,
                                                                                                                                                                                     labels, outliers_row_limit, nan_handlers)

print(f'The best parameters that yield a f1 score of {best_f1_score} are gamma = {best_gamma}, up_sampling_percentage = {best_up_sampling_percentage}, degree = {best_degree}, variance_treshold = {best_variance_threshold}, lambda = {best_lambda}, threshold = {best_threshold},max_iter = {best_max_iter}, best nan percentage = {best_nan_percentage}')
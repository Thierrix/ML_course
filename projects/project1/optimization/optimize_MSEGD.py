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

y, x = y_train.copy(), x_train.copy()
seed = 12
# Split data into k-fold indices
k_indices = build_k_indices(y, k_fold, seed)

# Initialize lists to store results
f1_score_array = []
param_combinations = []

# Generate all hyperparameter combinations using itertools.product
all_combinations = itertools.product(
    gammas,
    up_sampling_percentages,
    degrees,
    variances_threshold,
    lambdas,
    max_iters,
    decision_threshold,
    acceptable_nan_percentages,
    outliers_row_limit,
    nan_handlers,
)
# Total steps calculation and initialization
max_steps = (
    len(lambdas)
    * len(gammas)
    * len(up_sampling_percentages)
    * len(degrees)
    * len(variances_threshold)
    * len(max_iters)
    * len(decision_threshold)
    * len(acceptable_nan_percentages)
    * len(outliers_row_limit)
    * len(nan_handlers)
)
step = 1

# Iterate over each hyperparameter combination
for (
    gamma,
    up_sampling_percentage,
    degree,
    variance_threshold,
    lambda_,
    max_iter,
    threshold,
    acceptable_nan_percentage,
    outlier_limit,
    nan_handling,
) in all_combinations:
    total_f1_score_te = 0

    # Cross-validation loop
    for k in range(k_fold):
        # Perform cross-validation for the current fold
        f1_score_te = cross_validation(
            y,
            x,
            k_indices,
            k,
            lambda_,
            up_sampling_percentage,
            degree,
            variance_threshold,
            gamma,
            max_iter,
            threshold,
            acceptable_nan_percentage,
            labels,
            outlier_limit,
            nan_handling,
        )
        # Accumulate the F1-scores for test set
        total_f1_score_te += f1_score_te

    # Display the progress
    print(f"Step {step}/{max_steps}")
    step += 1

    # Average the F1-score over all folds
    avg_f1_score_te = total_f1_score_te / k_fold
    f1_score_array.append(avg_f1_score_te)

    # Store the corresponding parameter combination
    param_combinations.append(
        (
            gamma,
            up_sampling_percentage,
            degree,
            variance_threshold,
            lambda_,
            max_iter,
            threshold,
            acceptable_nan_percentage,
            outlier_limit,
            nan_handling,
        )
    )

    # Display the parameters for this iteration
    print(
        f"F1 score of {avg_f1_score_te} for gamma = {gamma}, up_sampling_percentage = {up_sampling_percentage}, degree = {degree}, variance_treshold = {variance_threshold}, lambda = {lambda_}, outlier limit = {outlier_limit}, max_iter = {max_iter}, threshold = {threshold}, nan percentage = {acceptable_nan_percentage}, nan handling = {nan_handling}"
    )

# Get the best F1 score and corresponding parameters
best_index = np.argmax(f1_score_array)
best_f1_score = f1_score_array[best_index]
best_params = param_combinations[best_index]

# Unpack the best parameters
(
    best_gamma,
    best_up_sampling_percentage,
    best_degree,
    best_variance_threshold,
    best_lambda,
    best_max_iter,
    best_threshold,
    best_nan_percentage,
    best_outlier_limit,
    best_nan_handler,
) = best_params

print(f'The best parameters that yield a f1 score of {best_f1_score} are gamma = {best_gamma}, up_sampling_percentage = {best_up_sampling_percentage}, degree = {best_degree}, variance_treshold = {best_variance_threshold}, lambda = {best_lambda}, threshold = {best_threshold},max_iter = {best_max_iter}, best nan percentage = {best_nan_percentage}')
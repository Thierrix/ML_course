import numpy as np
import matplotlib.pyplot as plt
from stats import f1_score
from cleaning_data import *
from implementations import *


def predict(tx, w, threshold):
    return np.where(sigmoid(tx @ w) >= threshold, 1, -1)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)


def cross_validation(
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
):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """

    # Test data from the k-th fold
    test_idx = k_indices[k]
    x_te = x[test_idx]
    y_te = y[test_idx]

    # Train data from the remaining k-1 folds
    train_idx = np.delete(np.arange(len(y)), test_idx)
    x_tr = x[train_idx]
    y_tr = y[train_idx]

    # Clean the training data
    (
        x_train_cleaned,
        y_tr_cleaned,
        features,
        median_and_most_probable_class,
        W,
        mean,
    ) = clean_train_data(
        x_tr,
        y_tr,
        labels,
        up_sampling_percentage,
        degree,
        variance_threshold,
        acceptable_nan_percentage,
        outlier_limit,
        nan_handling,
    )
    num_samples = x_train_cleaned.shape[0]
    tx_tr = np.c_[np.ones(num_samples), x_train_cleaned]

    # Process the testing data to put it in the same state as the training data
    x_te_cleaned = clean_test_data(
        x_te, labels, features, median_and_most_probable_class, mean, W, degree
    )
    num_samples = x_te_cleaned.shape[0]
    tx_te = np.c_[np.ones(num_samples), x_te_cleaned]

    # Initialize the initial weight vector, w
    mean = 0  # Mean of the distribution
    std_dev = 1  # Standard deviation of the distribution
    w_initial = np.random.normal(loc=mean, scale=std_dev, size=tx_tr.shape[1])
    # Train the model on this fold
    w, loss = reg_logistic_regression(
        y_tr_cleaned, tx_tr, lambda_, w_initial, max_iter, gamma
    )

    # Predict on the testing set for this fold
    y_predict = predict(tx_te, w, threshold)

    # Compute the testing F1 score of this fold
    f1_score_te = f1_score(y_te, y_predict)

    return f1_score_te


import itertools


def grid_search_k_fold(
    y,
    x,
    k_fold,
    lambdas,
    gammas,
    up_sampling_percentages,
    degrees,
    variances_threshold,
    max_iters,
    decision_threshold,
    acceptable_nan_percentages,
    labels,
    outliers_row_limit,
    nan_handlers,
):
    """
    Cross-validation over regularisation parameter lambda and other hyperparameters.

    Args:
        y, x: data and labels.
        k_fold: integer, the number of folds.
        lambdas, gammas, up_sampling_percentages, degrees, variances_threshold, max_iters,
        decision_threshold, acceptable_nan_percentages, outliers_row_limit, nan_handlers:
        Lists of hyperparameters for tuning.

    Returns:
        The best hyperparameter values and associated F1 score.
    """
    print("Beginning grid search with k fold cross validation")
    y, x = y.copy(), x.copy()
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
    print("Finished!")
    return (
        best_gamma,
        best_up_sampling_percentage,
        best_degree,
        best_variance_threshold,
        best_lambda,
        best_max_iter,
        best_f1_score,
        best_threshold,
        best_nan_percentage,
        best_outlier_limit,
        best_nan_handler,
    )

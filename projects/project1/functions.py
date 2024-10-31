import numpy as np
import matplotlib.pyplot as plt
from stats import f1_score
from cleaning_data import *
from implementations import *
import itertools

def predict(tx, w, threshold):
    """
    Predict the associated labels for every data points given the weights vector and the data points 
    Args:
        tx : numpy array of shape=(N,D + 1)
        w : numpy array of shape=(D+1, )
        threshold : scalar, the threshold for predicting cf. Prob(y_i = 1 / W,x_i)  > threshold -> y = 1
    Returns ;
        The predicted values, either 1 or -1 given the weights vector w, and the data points x 
    """
    return np.where(sigmoid(tx @ w) >= threshold, 1, -1)


def build_k_indices(y, k_fold, seed = 2):
    """build k indices for k-fold.
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)


def cross_validation(
    model,
    y,
    x,
    k_indices,
    k,
    labels,
    lambda_ = 0.01,
    up_sampling_percentage = 0.2,
    degree = 1,
    variance_threshold = 0.90,
    gamma = 0.5,
    max_iter = 300,
    threshold = 0.5,
    acceptable_nan_percentage = 1,
    outlier_limit = 1,
    nan_handling = 'mean',
):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        model: string specifying model used for learning
        y: numpy array of shape=(N,)
        x: numpy array of shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. penalized logistic regression
        up sampling percentage : The distribution we want for label 1 with respect to label - 1
        degree:     scalar, cf. build_poly()
        variance_threshold : scalar, cf. pca with this variance_threshold
        gamma : scalar, the learning rate
        max_iter : scalar, the max number of iterations
        threshold : scalar, the threshold for predicting cf Prob(y_i = 1 / W,x_i)  > threshold -> y = 1
        acceptable_nan_percentage : scalar, the percentage nan limit for features to be keeped
        outlier_limit : scalar, the outliers limit for rows to be keeped
        nan_handling : scalar, the method we use to replace the nan value
    Returns:
        The f1_score for this fold on the testing fold trained on the training fold
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
        mean_pca,
        mean,
        std_dev
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
        x_te, labels, features, median_and_most_probable_class, mean_pca, W, degree, mean, std_dev)
    num_samples = x_te_cleaned.shape[0]
    tx_te = np.c_[np.ones(num_samples), x_te_cleaned]

    # Initialize the initial weight vector, w
    mean = 0  # Mean of the distribution
    std_dev = 1  # Standard deviation of the distribution
    w_initial = np.random.normal(loc=mean, scale=std_dev, size=tx_tr.shape[1])

    # Train the model on this fold
    if model == "mean_squared_error_gd":
        w, loss = mean_squared_error_gd(
            y_tr_cleaned, tx_tr, w_initial, max_iter, gamma
        )
    elif model == "mean_squared_error_sgd":
        w, loss = mean_squared_error_sgd(
            y_tr_cleaned, tx_tr, w_initial, max_iter, gamma
        )
    elif model == "ridge_regression":
        w, loss = ridge_regression(
            y_tr_cleaned, tx_tr, lambda_
        )
    elif model == "least_squares":
        w, loss = least_squares(
            y_tr_cleaned, tx_tr
        )
    elif model == "logistic_regression":
        w, loss = logistic_regression(
            y_tr_cleaned, tx_tr, w_initial, max_iter, gamma
        )
    elif model == "reg_logistic_regression":
        w, loss = reg_logistic_regression(
            y_tr_cleaned, tx_tr, lambda_, w_initial, max_iter, gamma
        )

    # Predict on the testing set for this fold
    y_predict = predict(tx_te, w, threshold)
    # Compute the testing F1 score of this fold
    f1_score_te = f1_score(y_te, y_predict)
  
    return f1_score_te




import itertools
import numpy as np
import json

def grid_search_k_fold_logistic(
    models,
    y,
    x,
    k_fold,
    hyperparameters,
    labels
):
    """
    
    Grid search over hyperparameters for selected model.

    Args:
        model: string specifying model used for learning
        y: numpy array of shape=(N,)
        x: numpy array of shape=(N,D)
        k_fold: K in K-fold, i.e., the fold num
        hyperparameters: Dictionary of hyperparameters for tuning. Each key contains a list of values to search over.
        labels: Labels used in cross-validation.

    Returns:
        The best hyperparameter values and associated F1 score.
    """
    # Define default values if not provided in hyperparameters
    
    defaults = {
        "lambdas": [0.001],
        "gammas": [0.1],
        "up_sampling_percentages": [0.2],
        "degrees": [1],
        "variances_threshold": [0.99],
        "max_iters": [300],
        "decision_threshold": [0.5],
        "acceptable_nan_percentages": [1],
        "outliers_row_limit": [1],
        "nan_handlers": ["mean"]
    }
    
    # Update defaults with any provided hyperparameters
    for key in defaults:
        if key not in hyperparameters:
            hyperparameters[key] = defaults[key]

   
    y, x = y.copy(), x.copy()
    seed = 12
    # Split data into k-fold indices
    k_indices = build_k_indices(y, k_fold, seed)

    # Initialize lists to store results
    f1_score_array = []
    param_combinations = []

    # Generate all hyperparameter combinations using itertools.product
    all_combinations = itertools.product(
        models,
        hyperparameters["gammas"],
        hyperparameters["up_sampling_percentages"],
        hyperparameters["degrees"],
        hyperparameters["variances_threshold"],
        hyperparameters["lambdas"],
        hyperparameters["max_iters"],
        hyperparameters["decision_threshold"],
        hyperparameters["acceptable_nan_percentages"],
        hyperparameters["outliers_row_limit"],
        hyperparameters["nan_handlers"],
    )
    
    print('Grid creation completed')
    max_steps = len(list(itertools.product(
        *[hyperparameters[key] for key in hyperparameters]
    )))
    step = 1

    # Iterate over each hyperparameter combination
    for (
        model,
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
        print(f'Beginning grid search with k-fold cross-validation for {model}')
        total_f1_score_te = 0
        # Cross-validation loop
        for k in range(k_fold):
            # Perform cross-validation for the current fold
            f1_score_te = cross_validation(
                model,
                y,
                x,
                k_indices,
                k,
                labels,
                lambda_,
                up_sampling_percentage,
                degree,
                variance_threshold,
                gamma,
                max_iter,
                threshold,
                acceptable_nan_percentage,
                outlier_limit,
                nan_handling
            )
            total_f1_score_te += f1_score_te

        # Display the progress
        print(f"Step {step}/{max_steps}")
        step += 1

        # Average the F1-score over all folds
        avg_f1_score_te = total_f1_score_te / k_fold
        f1_score_array.append(avg_f1_score_te)

        # Store the corresponding parameter combination
        current_params = {
            "model": model,
            "gamma": gamma,
            "up_sampling_percentage": up_sampling_percentage,
            "degree": degree,
            "variance_threshold": variance_threshold,
            "lambda_": lambda_,
            "max_iter": max_iter,
            "threshold": threshold,
            "acceptable_nan_percentage": acceptable_nan_percentage,
            "outlier_limit": outlier_limit,
            "nan_handling": nan_handling
        }
        
        param_combinations.append(current_params)

        # Save current parameters to a JSON file
        with open("hyperparameters_iteration.json", "a") as file:
            json.dump({"iteration": step - 1, "params": current_params, "f1_score": avg_f1_score_te}, file, indent=4)
            file.write("\n")

        # Display the parameters for this iteration
        print(
            f'F1 score of {avg_f1_score_te}, gamma = {gamma}, up_sampling_percentage = {up_sampling_percentage}, '
            f'degree = {degree}, variance_threshold = {variance_threshold}, lambda = {lambda_}, '
            f'outlier limit = {outlier_limit}, max_iter = {max_iter}, threshold = {threshold}, '
            f'nan percentage = {acceptable_nan_percentage}, nan handling = {nan_handling}'
        )

    # Get the best F1 score and corresponding parameters
    best_index = np.argmax(f1_score_array)
    best_f1_score = f1_score_array[best_index]
    best_params = param_combinations[best_index]

    # Save the best parameters to a JSON file
    with open("best_hyperparameters.json", "w") as file:
        json.dump({"best_params": best_params, "best_f1_score": best_f1_score}, file, indent=4)

    print("Finished! Best hyperparameters saved to best_hyperparameters.json")
    return best_params

def slice_data(x, y, num_slices, seed):    
    """
    Determine the size of each slice for x and y

    Args :
        x : numpy array of shape=(N,D)
        y : numpy array of shape=(N,)
        num_slices : scalar, the number of slices to do on x and y
        seed:  the random seed
    Returns :
        A array containings the slices for both x and y
    
    """
     # Shuffle indices for training data
    np.random.seed(seed)
    train_indices = np.random.permutation(len(x))

    x_train_shuffled = x[train_indices]
    y_train_shuffled = y[train_indices]
    train_slice_size = len(x) // num_slices
    x_train_slices = [x_train_shuffled[i * train_slice_size: (i + 1) * train_slice_size] for i in range(num_slices)]
    y_train_slices = [y_train_shuffled[i * train_slice_size: (i + 1) * train_slice_size] for i in range(num_slices)]
    
    
    return x_train_slices, y_train_slices

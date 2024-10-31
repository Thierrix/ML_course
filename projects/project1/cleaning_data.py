import numpy as np
import matplotlib.pyplot as plt
from helpers import *



# Defining some constants
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from utils import (
    remove_features,
    find_key_by_value,
    create_pca,
    apply_pca_given_components,
    upsample_class_1_to_percentage,
    build_poly,
)
from stats import z_score_normalization, min_max_normalization, normalize_data, IQR



def clean_train_data(
    x_train,
    y_train,
    labels,
    up_sampling_percentage,
    degree,
    variance_threshold,
    acceptable_nan_percentage,
    outliers_row_limit,
    nan_handling="median",
):
    """
    Cleaning data
    :param x_train: training data
    :return: cleaned data
    """
    x = x_train.copy()
    y = y_train.copy()

    median_and_most_probable_class = {}
    y[y == -1] = 0
    

    # Removing the first label which is the id
    features_number = x.shape[1]
    features = list(dict.fromkeys(labels))
    features = [features[i] for i in range(features_number)]
    features = {word: index for index, word in enumerate(features)}

    # Removing columns with more than ACCEPTABLE_NAN_PERCENTAGE of NaN values
    mask_nan_columns = [
        (np.count_nonzero(np.isnan(x[:, i])) / x.shape[0]) <= acceptable_nan_percentage
        for i in range(features_number)
    ]
    x = x[:, mask_nan_columns]

    # Creating features list
    features = list(dict.fromkeys(labels))
    features = [features[i] for i in range(features_number) if mask_nan_columns[i]]
    features = {word: index for index, word in enumerate(features)}

    # We remove the features that are not useful

    x = handle_nan(x, features, median_and_most_probable_class, nan_handling)

    # Remove outliers before PCA to avoid their effect on components
    x, y = remove_outliers(x, y, outliers_row_limit)
    x, y = upsample_class_1_to_percentage(x, y, up_sampling_percentage)
    # Polynomial feature expansion before PCA
    x = build_poly(x, degree)
    
    # Normalize the data
    x, mean, std_dev = normalize_data(x)
    
    # Create a PCA on the training set, storing the components and mean of this PCA in order to replicate it
    x, W, mean_pca = create_pca(x, variance_threshold)
    
    # Upsample class 1 to the required percentage


    return x, y, features, median_and_most_probable_class, W, mean_pca, mean, std_dev

def clean_test_data(
    x_te, labels, features, median_and_most_probable_class, mean_pca, W, degree, mean, std_dev
):

    x = x_te.copy()
    # Keep only the features we kept on the training set
    mask = [feature in features.keys() for feature in labels]
    x = x[:, mask]

    # Replace the nan in the same way we did on the training set
    for feature in features:
        nan = median_and_most_probable_class[feature]
        x[:, features[feature]] = np.nan_to_num(
            x[:, features[feature]], nan=median_and_most_probable_class[feature]
        )

    # Normalize the testing data independtly from the training set
    x = build_poly(x, degree)
    x = (x - mean) / std_dev
    # Apply the pca given the training PCA
    x = apply_pca_given_components(x, mean_pca, W)
    return x


def handle_nan(x, features, median_and_most_probable_class, nan_handling):

    # Replace NaN in categorical features with the median value
    for feature in features:
        if nan_handling == "median":
            replace_value = np.nanmedian(x[:, features[feature]])

        elif nan_handling == "mean":
            replace_value = np.nanmean(x[:, features[feature]])

        elif nan_handling == "numeric":
            replace_value = -1

        #Store the median to replace it on the testing data
        median_and_most_probable_class[feature] = replace_value
        x[:, features[feature]] = np.nan_to_num(
            x[:, features[feature]], nan=replace_value
        )

    return x

def remove_outliers(x, y, outliers_row_limit):
    
    lower_bound, upper_bound = IQR(x)
    outlier = (x < lower_bound) | (x > upper_bound)
    outlier_number = np.sum(outlier, axis=1)
    mask = outlier_number <= outliers_row_limit * x.shape[1]

    return x[mask, :], y[mask]
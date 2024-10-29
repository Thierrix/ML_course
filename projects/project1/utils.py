import numpy as np


def apply_pca_given_components(X, mean, W):
    """
    Apply the PCA transformation from a specific PCA.
    Args :
        X: data to apply PCA on
        mean : the mean of the PCA to apply
        W : the principal components of the PCA to apply
    Returns :
        x_pca : PCA applied on the data
    """
    # Subtract by the mean
    x_tilde = X - mean

    # Project the data onto the principal components
    x_pca = np.dot(x_tilde, W)

    return x_pca


def create_pca(X, variance_threshold=0.90):
    """
    Create a PCA given the data, either retaining a specified percentage of variance or specifying the number of components.

    Args :
        X : data
        variance_threshold: the desired amount of variance to retain (default 90%)
        num_components: the desired number of principal components (default None, use variance threshold)

    Returns :
        x_pca : PCA applied on the data, X
        W : the principal components of the PCA
        mean : the mean of x_train

    """

    mean = np.nanmean(X, axis=0)
    x_tilde = X - mean

    cov_matrix = np.cov(x_tilde, rowvar=False)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals = eigvals[::-1]  # Sort eigenvalues in descending order
    eigvecs = eigvecs[:, ::-1]  # Sort eigenvectors accordingly

    # Calculate cumulative variance explained by the principal components
    explained_variance_ratio = eigvals / np.sum(eigvals)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Determine the number of components to retain based on the variance threshold
    num_dimensions = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Select the top principal components
    W = eigvecs[:, :num_dimensions]  # Select top components based on desired condition

    # Project the data onto the selected principal components
    x_pca = np.dot(x_tilde, W)

    return x_pca, W, mean


def upsample_class_1_to_percentage(X, y, desired_percentage):
    """
    Apply a upsampling to obtain the desired percentage of 1 among the data.

        Args :
            X: X data to upsample
            y : y data to usample
            desired_precentage : the desired repartition of 1 among the data, must be higher than the initial percentage and lower than 1
        Returns :
            X_upsampled : X upsampled to attain the desired percentage
            y_upsampled : y upsampled to attain the desired percentage
    """
    # Find the indices of class 0 (majority class) and class 1 (minority class)
    initial_percentage = np.sum(y == 1)/y.shape[0]
    assert(desired_percentage >= initial_percentage and desired_percentage <= 1), "The desired percentage must be higher than the initial percentage"

    indices_class_1 = np.where(y == 1)[0]
    indices_class_0 = np.where(y == 0)[0]

    # Number of samples in each class
    num_class_1 = len(indices_class_1)
    num_class_0 = len(indices_class_0)

    # Calculate the total number of samples needed for the desired percentage
    total_size = int(num_class_0 / (1 - desired_percentage))

    # Calculate the number of class 1 samples needed to reach the desired percentage
    target_num_class_1 = int(total_size * desired_percentage)

    # Upsample class 1 by randomly duplicating samples until reaching the target number
    upsampled_indices_class_1 = np.random.choice(
        indices_class_1, size=target_num_class_1, replace=True
    )

    # Combine the upsampled class 1 samples with all class 0 samples
    combined_indices = np.concatenate([indices_class_0, upsampled_indices_class_1])

    # Shuffle the combined dataset to avoid ordering bias
    np.random.shuffle(combined_indices)

    # Return the upsampled feature matrix and label vector
    X_upsampled = X[combined_indices]
    y_upsampled = y[combined_indices]

    return X_upsampled, y_upsampled


def downsample_class_0(X, y, desired_percentage):
    """
    Apply a downsampling to obtain the desired percentage of 0 among the data.

    Args :
        X: data to upsample
        y :
        desired_precentage : the desired repartition of 1 among the data
    Returns :
        X_upsampled : X upsampled to attain the desired percentage
        y_upsampled : y upsampled to attain the desired percentage

    """
    N_pos = len(np.where(y == 1)[0])
    N_neg = len(np.where(y == 0)[0])
    down_sample_size_0 = calculate_downsample_size(N_pos, N_neg, desired_percentage)
    # Separate the samples with label 0 and label 1
    indices_class_0 = np.where(y == 0)[0]
    indices_class_1 = np.where(y == 1)[0]

    # Downsample the class 0 samples
    selected_indices_class_0 = np.random.choice(
        indices_class_0, size=downsample_size_0, replace=False
    )

    # Combine the downsampled class 0 samples with all class 1 samples
    combined_indices = np.concatenate([selected_indices_class_0, indices_class_1])

    # Shuffle the combined indices to mix the samples
    np.random.shuffle(combined_indices)

    # Select the corresponding rows from X and y
    X_downsampled = X[combined_indices]
    y_downsampled = y[combined_indices]

    return X_downsampled, y_downsampled


def remove_features(x, features_to_remove, features):
    """
    Remove the selected features from x and from the available features list.

    Args :
        x:  numpy array of shape (N,D), N is the number of samples and D the number of features.
        features_to_reùpve : the desired repartition of 1 among the data
    Returns :
        X_upsampled : X upsampled to attain the desired percentage
        y_upsampled : y upsampled to attain the desired percentage

    """
    if not set(features_to_remove).intersection(features):
        return features, x

    for feature in features_to_remove:
        if feature in features:
            x = np.delete(x, features[feature], axis=1)
            features = remove_and_update_indices(features, feature)

    return features, x


def find_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None


def remove_and_update_indices(d, remove_key):
    if remove_key in d:
        removed_index = d.pop(remove_key)
        for key in d:
            if d[key] > removed_index:
                d[key] -= 1

    return d


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.


    """

    # set seed

    np.random.seed(seed)
    x = np.random.permutation(x)
    np.random.seed(seed)
    y = np.random.permutation(y)
   
    sample_size = y.shape[0]
    training_size = (int)(np.floor(ratio * sample_size))
    x_tr = x[:training_size]
    x_te = x[training_size:]
    y_tr = y[:training_size]
    y_te = y[training_size:]

    return x_tr, x_te, y_tr, y_te


def calculate_downsample_size(N_pos, N_neg, desired_percentage):
    # Calculate the required downsample size for the negative class
    downsample_size_neg = int((N_pos * (1 - desired_percentage)) / desired_percentage)

    # Ensure we do not downsample more than available negative samples
    downsample_size_neg = min(downsample_size_neg, N_neg)

    return downsample_size_neg


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x (np.ndarray, (N,)): array of shape, N is the number of samples.
        degree (int, optional): degree of the polynomial. Defaults to 1.

    Returns:
        poly (np.ndarray, (N,D+1)): the computed polynomial features.
    """
    poly = None
    for deg in range(1, degree + 1):
        if poly is None:
            poly = np.power(x, deg)
        else:
            poly = np.c_[poly, np.power(x, deg)]
    return poly


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,D), N is the number of samples and D the number of features
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)
    """
    N, D = x.shape
    poly = np.ones((N, (degree + 1) * D))  # Initialize polynomial feature matrix

    # Expand each feature in X to its powers from 0 to the given degree
    for i in range(D):
        for j in range(degree + 1):
            poly[:, i * (degree + 1) + j] = np.power(x[:, i], j)

    return poly


import numpy as np
import matplotlib.pyplot as plt
from stats import f1_score
from clean_data_testing import *
from cleaning_data import *
import time


def predict(tx ,w, threshold) :
    return np.where(sigmoid(tx@w) >= threshold, 1, -1)


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """

    t = np.clip(t, -709, 709)  # Limit the value of t to avoid overflow
    return 1 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    #add an espilon to ensure that if there is a value very close to 0 it doesn't do log of a zero value
    epsilon = 1e-15

    #compute the negative log loss
    sigmoid_output = sigmoid(tx@w)
    negative_log_loss = -np.mean(y * np.log(np.clip(sigmoid_output, epsilon, 1 - epsilon)) + (1 - y) * np.log(np.clip(1 - sigmoid_output, epsilon, 1 - epsilon)))
    
    return negative_log_loss



def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    return (1/y.shape[0])*tx.T@(sigmoid(tx@w) - y)


    
def logistic_regression(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> loss, gradient, hessian = logistic_regression(y, tx, w)
    >>> round(loss, 8)
    0.62137268
    >>> gradient, hessian
    (array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]]), array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]]))
    """
    return calculate_loss(y, tx, w), calculate_gradient(y, tx, w)


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    loss, gradient = logistic_regression(y, tx, w)

    loss = loss + lambda_*np.dot(w, w)
    gradient = gradient + 2*lambda_*w
    
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> gamma = 0.1
    >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    loss, gradient = penalized_logistic_regression(y, tx, w,lambda_)

    w = w - gamma*gradient
    return loss, w
    
def stochastic_gradient_descent(y, tx ,initial_w, batch_size, max_iters, gamma, lambda_, model = 'logistic'):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    f1_tr = []
    losses = []
    w = initial_w
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y,tx,batch_size) :
            
            if(model == 'logistic') :
                loss, w = learning_by_penalized_gradient(minibatch_y, minibatch_tx, w, gamma, lambda_)
            elif(model == 'lstq') :
                continue
            losses.append(loss)
            ws.append(w)
            
            
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
        
    return losses, ws
    


def gradient_descent(y, tx ,initial_w, max_iters, gamma, lambda_, model = 'logistic'):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        if(model == 'logistic') :
            loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        elif(model == 'lstq') :
            continue
        losses.append(loss)
        ws.append(w) 
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
                
    return losses, ws








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


def cross_validation(y, x, k_indices, k, lambda_, up_sampling_percentage, degree, variance_threshold, gamma, max_iter, threshold,acceptable_nan_percentage,
                     labels, outlier_limit, nan_handling):
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

    #Test data from the k-th fold
    test_idx = k_indices[k]
    x_te = x[test_idx]
    y_te = y[test_idx]
    
    # Train data from the remaining k-1 folds
    train_idx = np.delete(np.arange(len(y)), test_idx)
    x_tr = x[train_idx]
    y_tr = y[train_idx]


    #Clean the training data
    x_train_cleaned, y_tr_cleaned, features, median_and_most_probable_class, W, mean = clean_train_data(x_tr, y_tr,labels,up_sampling_percentage , degree, variance_threshold,     acceptable_nan_percentage, outlier_limit, nan_handling)
    num_samples = x_train_cleaned.shape[0]
    tx_tr = np.c_[np.ones(num_samples), x_train_cleaned]

    #Process the testing data to put it in the same state as the training data
    x_te_cleaned = clean_test_data(x_te, labels, features, median_and_most_probable_class, mean, W, degree)
    num_samples = x_te_cleaned.shape[0]
    tx_te = np.c_[np.ones(num_samples), x_te_cleaned]

    #Initialize the initial weight vector, w
    mean = 0    # Mean of the distribution
    std_dev = 1 # Standard deviation of the distribution
    w_initial = np.random.normal(loc=mean, scale=std_dev, size=tx_tr.shape[1])

    #Train the model on this fold
    losses, ws = gradient_descent(
    y_tr_cleaned, tx_tr, w_initial, max_iter, gamma, lambda_
    )
    
    #Predict on the testing set for this fold
    y_predict = predict(tx_te, ws[-1], threshold)

    #Compute the testing F1 score of this fold
    f1_score_te = f1_score(y_te, y_predict)

    return f1_score_te




def cross_validation_demo(y, x, k_fold, lambdas, gammas, up_sampling_percentages, degrees, variances_threshold,max_iters,
                          decision_threshold, acceptable_nan_percentages,labels, outliers_row_limit, nan_handlers):
    """cross validation over regularisation parameter lambda.

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    y = y.copy()
    x = x.copy()
    seed = 12
    # Split data into k-fold indices
    k_indices = build_k_indices(y, k_fold, seed)
    
    # Initialize lists to store results
    f1_score_array = []
    param_combinations = []
    max_steps = len(lambdas)*len(gammas)*len(up_sampling_percentages)*len(degrees)*len(variances_threshold)*len(max_iters)*len(decision_threshold)*len(acceptable_nan_percentages)*len(outliers_row_limit)*len(nan_handlers)
    step = 1
    
    # Iterate over all hyperparameters
    for nan_handling in nan_handlers : 
        for acceptable_nan_percentage in acceptable_nan_percentages :
            for threshold in decision_threshold :
                for gamma in gammas:
                    for up_sampling_percentage in up_sampling_percentages:
                        for degree in degrees:
                            for variance_threshold in variances_threshold:
                                for lambda_ in lambdas:
                                    for max_iter in max_iters : 
                                        for outlier_limit in outliers_row_limit :
                                            
                                            total_f1_score_te = 0
                                            # Cross-validation loop
                                            for k in range(k_fold):
                                                # Perform cross-validation for the current fold
                                                f1_score_te = cross_validation(y, x, k_indices, k, lambda_, up_sampling_percentage, degree, variance_threshold, gamma,max_iter, threshold,acceptable_nan_percentage,labels, outlier_limit, nan_handling)
                                                # Accumulate the F1-scores for test set
                                                total_f1_score_te += f1_score_te
                                                
                                            #Display at which step we are compaired to the total steps number
                                            print(f'Step {step}/{max_steps}')
                                            step += 1
                                            
                                            # Average the F1-score over all folds
                                            avg_f1_score_te = total_f1_score_te / k_fold
                                            f1_score_array.append(avg_f1_score_te)
                                            
                                            # Store the corresponding parameter combination
                                            param_combinations.append((gamma, up_sampling_percentage, degree, variance_threshold, lambda_, max_iter, threshold, acceptable_nan_percentage, outlier_limit, nan_handling))
                                            #Display the parameters for this iteration
                                            print(f'F1 score of {avg_f1_score_te} for gamma = {gamma}, up_sampling_percentage = {up_sampling_percentage}, degree = {degree}, variance_treshold = {variance_threshold}, lambda = {lambda_}, outlier limit = {outlier_limit} max_iter = {max_iter}, threshold = {threshold}, nan percentage = {acceptable_nan_percentage}, nan handling = {nan_handling}')
                                            
    # Get the best F1 score and corresponding parameters
    best_index = np.argmax(f1_score_array)
    best_f1_score = f1_score_array[best_index]
    best_params = param_combinations[best_index]
    
    # Unpack the best parameters
    best_gamma, best_up_sampling_percentage, best_degree, best_variance_threshold, best_lambda, best_max_iter, best_threshold, best_nan_percentage, best_outlier_limit, best_nan_handler = best_params
    print('finished !')
    return best_gamma, best_up_sampling_percentage, best_degree, best_variance_threshold, best_lambda,best_max_iter, best_f1_score, best_threshold, best_nan_percentage, best_outlier_limit, best_nan_handler
   
    
    


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.ÅÅÅÅ
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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







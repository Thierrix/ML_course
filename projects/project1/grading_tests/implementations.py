import numpy as np
from utils2 import batch_iter

# compute the mean squared error of a model
def MSE(y, tx, w):
    """Computes the mean squared error at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A scalar containing the error of the loss at w.
    """
    N = y.shape[0]
    e = y - (tx @ w)
    loss = 1/(2*N) * np.sum(np.power(e, 2))

    return loss

# compute the gradient of the MSE loss function
def compute_gradient_MSE(y, tx, w):
    """Computes the mean squared error gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    e = y - (tx @ w)
    grad = -1/N * (tx.T @ e)
    return grad

# compute the regularized MSE loss, return both the non-regularized and the regularized loss
def MSE_regularized(y, tx, w, lambda_):
    """Computes the regularized mean squared error at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        lambda_ : scalare. The regulaeization coefficient
    Returns:
        A scalar containing the error of the loss at w.
    """
    loss = MSE(y, tx, w)
    regularizer = lambda_*(np.linalg.norm(w)**2)
    return loss, loss + regularizer

# compute sigmoid function
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
    return np.exp(t)/(1 + np.exp(t))

# compute negative log likelihood loss of a model
def neg_log_loss(y, tx, w):
    prob = sigmoid(tx @ w)
    epsilon = 1e-15
    return -np.mean(y * np.log(np.clip(prob, epsilon, 1)) + (1 - y) * np.log(np.clip(1 - prob, epsilon, 1)))

# compute gradient of negative log likelihood loss function
def neg_log_gradient(y, tx, w):
    gradient = 1/y.shape[0] * (tx.T @ (sigmoid(tx @ w) - y))
    return gradient

# compute gradient of regularized negative log likelihood loss function
def neg_log_gradient_reg(y, lambda_, tx, w):
    gradient = neg_log_gradient(y, tx, w)
    return gradient + 2*lambda_*w

# train model using least squares 
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    N = y.shape[0]
    y = np.reshape(y, (N,))
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    MSE = 1/(2*N) * np.sum((y - (tx @ w))**2, 0)
    return w, MSE

# train model using gradient descent on the MSE loss function
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """"The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.        
    """

    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8 # define convergence when difference between losses of two consecutive iters falls below this
    ws = [initial_w]
    initial_loss =  MSE(y, tx, initial_w)
    losses = [initial_loss]
    w = initial_w
    
    for n_iter in range(max_iters): # iterating on guess for model weights
        gradient = compute_gradient_MSE(y, tx, w) # compute sgradient over all data points
        w -= gamma*gradient # update w based on gradient computation
        loss = MSE(y, tx, w)
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence achieved, end loop and return weights/loss

    return w, losses[-1]

# train model using gradient descent on the MSE loss function
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: A scalar denoting the stepsize
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.

    """
    
    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8 # define convergence when difference between losses of two consecutive iters falls below this
    ws = [initial_w]
    initial_loss =  MSE(y, tx, initial_w)
    losses = [initial_loss]
    w = initial_w
    for n_iter in range(max_iters): # iterating on guess for model weights
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1): # iterating for batch subset of data points
            gradient = compute_gradient_MSE(minibatch_y, minibatch_tx, w) # compute gradient from subset
            w -= gamma*gradient # update weights with gradient
            loss = MSE(minibatch_y, minibatch_tx, w)
            ws.append(w)
            losses.append(loss)

            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break # convergence achieved, end loop and return weights/loss

    return w, losses[-1]

# train model using ridge regression
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N = y.shape[0]
    y = np.reshape(y, (N,))
    D = np.shape(tx)[1]
    lambda_prime = 2*N*lambda_  
    w = np.linalg.solve(((tx.T @ tx) + lambda_prime * np.identity(D)), (tx.T @ y))
    MSE, MSE_reg = MSE_regularized(y, tx, w, lambda_)
    return w, MSE

# train model using logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
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

    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8 # define convergence when difference between losses of two consecutive iters falls below this
    w = initial_w
    initial_loss =  neg_log_loss(y, tx, initial_w)
    losses = [initial_loss]
   
    
    for n_iter in range(max_iters): # iterating on guess for model weights

        gradient = neg_log_gradient(y, tx, w) # compute gradient of negative log likelihood loss at model weights
        w -= gamma*gradient # updating the weights based on gradient
        loss = neg_log_loss(y, tx, w)
        losses.append(loss)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence achieved, end loop and return weights/loss

    return w, losses[-1]

# train model using regularized logistic regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
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
    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8 # define convergence when difference between losses of two consecutive iters falls below this
    w = initial_w
    initial_loss =  neg_log_loss(y, tx, initial_w)
    losses = [initial_loss]
   
    
    for n_iter in range(max_iters): # iterating on guess for model weights

        gradient = neg_log_gradient_reg(y, lambda_, tx, w) # compute gradient of regularized negative log likelihood
        w -= gamma*gradient # update model weights based on gradient
        loss = neg_log_loss(y, tx, w)
        losses.append(loss)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence achieved, end loop and return weights/loss

    return w, losses[-1]

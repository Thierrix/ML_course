import numpy as np
from utils import batch_iter


def MSE(y, tx, w):
    """Computes the mean squared error at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A scalar containing the error of the loss at w.
    """
    N = y.shape[0]
    e = y - (tx @ w)
    loss = 1 / (2 * N) * np.sum(np.power(e, 2))

    return loss



def compute_gradient_MSE(y, tx, w):
    """Computes the mean squared error gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    e = y - (tx @ w)
    grad = -1 / N * (tx.T @ e)
    return grad



def MSE_regularized(y, tx, w, lambda_):
    """
    compute the regularized MSE loss, return both the non-regularized and the regularized loss

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        w: numpy array of shape=(D, ). The vector of model parameters.
        lambda_ : scalar. The regularization coefficient
    Returns:
        A scalar containing the error regularized and non-regularized of the loss at w.
    """
    loss = MSE(y, tx, w)
    regularizer = lambda_ * (np.linalg.norm(w) ** 2)
    return loss, loss + regularizer


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
   # Use a numerically stable sigmoid function to avoid overflow
    return np.where(
        t >= 0,
        1 / (1 + np.exp(-t)),
        np.exp(t) / (1 + np.exp(t))
    )


# compute negative log likelihood loss of a model
def neg_log_loss(y, tx, w):
    """
    compute the negative log loss at w 
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        w: numpy array of shape=(D, ). The vector of model parameters.
    Returns:
        A scalar containing the error of the loss at w.
    """
    prob = sigmoid(tx @ w)

    #small value epsilon to avoid overflow
    epsilon = 1e-15
    return -np.mean(
        y * np.log(np.clip(prob, epsilon, 1))
        + (1 - y) * np.log(np.clip(1 - prob, epsilon, 1))
    )



def neg_log_gradient(y, tx, w):
    """
    compute the negative log gradient at w
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        w: numpy array of shape=(D, ). The vector of model parameters.
    Returns:
        A numpy array of same shape as w containing the gradient of the loss at w.
    """
    gradient = 1 / y.shape[0] * (tx.T @ (sigmoid(tx @ w) - y))
    return gradient


# compute gradient of regularized negative log likelihood loss function
def neg_log_gradient_reg(y, lambda_, tx, w):
    gradient = neg_log_gradient(y, tx, w)
    return gradient + 2 * lambda_ * w



def least_squares(y, tx):
    """Calculate the least squares solution.
       returns optimal weights and mse

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D+1), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D+1,), D is the number of features.
        mse: scalar.
    """
    N = y.shape[0]
    y = np.reshape(y, (N,))
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = 1 / (2 * N) * np.sum((y - (tx @ w)) ** 2, 0)
    return w, mse



def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """ "The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        initial_w: numpy array of shape=(D+1, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the learning rate

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8  # define convergence when difference between losses of two consecutive iters falls below this
    ws = [initial_w]
    initial_loss = MSE(y, tx, initial_w)
    losses = [initial_loss]
    w = initial_w

    for n_iter in range(max_iters):  # iterating on guess for model weights
        gradient = compute_gradient_MSE(
            y, tx, w
        )  # compute sgradient over all data points
        w -= gamma * gradient  # update w based on gradient computation
        loss = MSE(y, tx, w)
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break  # convergence achieved, end loop and return weights/loss

    return w, losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D+1, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: A scalar denoting the learning rate
    Returns:
        w: optimal weights, numpy array of shape(D+1,), D is the number of features.
        loss: scalar.

    """

    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8  # define convergence when difference between losses of two consecutive iters falls below this
    ws = [initial_w]
    initial_loss = MSE(y, tx, initial_w)
    losses = [initial_loss]
    w = initial_w
    for n_iter in range(max_iters):  # iterating on guess for model weights
        for minibatch_y, minibatch_tx in batch_iter(
            y, tx, 1
        ):  # iterating for batch subset of data points
            gradient = compute_gradient_MSE(
                minibatch_y, minibatch_tx, w
            )  # compute gradient from subset
            w -= gamma * gradient  # update weights with gradient
            loss = MSE(minibatch_y, minibatch_tx, w)
            ws.append(w)
            losses.append(loss)

            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break  # convergence achieved, end loop and return weights/loss

    return w, losses[-1]


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D+1), D is the number of features.
        lambda_: scalar, the regularization coefficient.

    Returns:
        w: optimal weights, numpy array of shape(D+1,), D is the number of features.
    """
    N = y.shape[0]
    y = np.reshape(y, (N,))
    D = np.shape(tx)[1]
    lambda_prime = 2 * N * lambda_
    w = np.linalg.solve(((tx.T @ tx) + lambda_prime * np.identity(D)), (tx.T @ y))
    MSE, MSE_reg = MSE_regularized(y, tx, w, lambda_)
    return w, MSE


# train model using logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D+1)
        w:  shape=(D+1, 1)
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: A scalar denoting the learning rate

    Returns:
        w, loss where w is a numpy array of shape (D+1,) and loss a scalar
    """

    N = y.shape[0]
    # y = np.reshape(y, (N,))
    threshold = 1e-8  # define convergence when difference between losses of two consecutive iters falls below this
    w = initial_w
    initial_loss = neg_log_loss(y, tx, initial_w)
    losses = [initial_loss]

    for n_iter in range(max_iters):  # iterating on guess for model weights

        gradient = neg_log_gradient(
            y, tx, w
        )  # compute gradient of negative log likelihood loss at model weights
        w -= gamma * gradient  # updating the weights based on gradient
        loss = neg_log_loss(y, tx, w)
        losses.append(loss)

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break  # convergence achieved, end loop and return weights/loss

    return w, losses[-1]


# train model using regularized logistic regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D+1)
        w:  shape=(D+1, 1)
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the learning rate
        lambda_: scalar, scalar, the regularization coefficient

    Returns:
        w: shape=(D + 1, 1)
        loss: scalar number

    """
    N = y.shape[0]
    y = np.reshape(y, (N,))
    threshold = 1e-8  # define convergence when difference between losses of two consecutive iters falls below this
    w = initial_w
    initial_loss = neg_log_loss(y, tx, initial_w)
    losses = [initial_loss]

    for n_iter in range(max_iters):  # iterating on guess for model weights

        gradient = neg_log_gradient_reg(
            y, lambda_, tx, w
        )  # compute gradient of regularized negative log likelihood
        w -= gamma * gradient  # update model weights based on gradient
        loss = neg_log_loss(y, tx, w)
        losses.append(loss)

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break  # convergence achieved, end loop and return weights/loss

    return w, losses[-1]

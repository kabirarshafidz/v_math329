import numpy as np
import time


def quad_smooth(x):
    """
    Evaluates the piecewise quadratic smoothed hinge loss phi(x).
    
    phi(x) = 
        0                   if x < -1
        0.5 * (1 + x)^2     if -1 <= x <= 0
        x + 0.5             if x > 0
    """
    x = np.array(x, dtype='float')
    z = np.zeros_like(x)
    # Quadratic transition region: smooth approximation between flat and linear parts
    mask_mid = (x >= -1) & (x <= 0)
    z[mask_mid] = 0.5 * (1 + x[mask_mid]) ** 2
    # Linear region: slope of 1
    z[x > 0] = x[x > 0] + 0.5

    return z


def qs_grad(x): 
    """
    Evaluates the derivative of the smoothed hinge loss phi'(x).
    
    phi'(x) = 
        0       if x < -1
        1 + x   if -1 <= x <= 0
        1       if x > 0
    """
    x = np.array(x, dtype='float')
    z = np.zeros_like(x)
    # Derivative in the quadratic region
    mask_mid = (x >= -1) & (x <= 0)
    z[mask_mid] = 1 + x[mask_mid]
    # Derivative in the linear region
    z[x > 0] = 1

    return z


def funcV(theta, x_dat, y_label, lam=0.005):
    """
    Computes the regularized objective value using vectorized NumPy operations.
    """
    # Vector of signs s in {+1, -1}^n
    s = 1 - 2 * y_label
    # Margins for all samples computed simultaneously: s * (X @ theta)
    margins = s * (x_dat @ theta)
    # Sum of losses plus L2 regularization term
    return np.sum(quad_smooth(margins)) + (lam / 2.0) * np.linalg.norm(theta) ** 2


def f_gradV(theta, x_dat, y_label, lam=0.005):
    """
    Computes the gradient of the objective using vectorized NumPy operations.
    """
    s = 1 - 2 * y_label
    margins = s * (x_dat @ theta)
    # Scalar weight per sample: s_i * phi'(margins_i)
    weights = qs_grad(margins) * s
    # Gradient: X.T @ weights + lam * theta
    return lam * theta + x_dat.T @ weights
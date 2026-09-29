import os
import numpy as np
from Q4 import data_loader


def predict(theta, X):
    """
    Predicts binary digit labels {0, 1} for a given data matrix X.
    
    Decision rule derived from the smoothed hinge loss formulation:
      s_i = 1 - 2*y_i maps class 0 -> +1 and class 1 -> -1.
      Loss decreases when s_i * <x_i, theta> >= 0.
      Therefore:
        <x_i, theta> >= 0 predicts label 0.
        <x_i, theta> < 0  predicts label 1.
    """
    scores = X @ theta
    return (scores >= 0).astype(int)


def error_rate(theta, X, y):
    """
    Computes the 0-1 classification error rate:
        Error = (number of misclassified samples) / (total samples)
    """
    preds = predict(theta, X)
    return float(np.mean(preds != y))


def q7_error_check(path, final_theta, results_dir='../results'):
    """
    Evaluates final iterate theta on train and test splits, reports numerical error
    rates, and saves the values as requested by Question 7.
    """
    # Load training and test splits
    X_train, y_train, X_test, y_test = data_loader(path)

    train_err = error_rate(final_theta, X_train, y_train)
    test_err = error_rate(final_theta, X_test, y_test)

    print(f"train error rate = {train_err:.5f} ({train_err * 100:.2f}%)")
    print(f"test error rate  = {test_err:.5f} ({test_err * 100:.2f}%)")

    # Save output using robust cross-platform path resolution
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, 'q7_error_rates.txt')
    with open(out_file, 'w') as file:
        file.write(f"train_error_rate={train_err}\n")
        file.write(f"test_error_rate={test_err}\n")

    return train_err, test_err

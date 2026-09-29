import numpy as np
from Q4 import data_loader


def predict(theta, X):
    scores = X @ theta
    return (scores >= 0).astype(int)


def error_rate(theta, X, y):
    preds = predict(theta, X)
    return np.mean(preds != y)


def q7_error_check(path, final_theta):
    X_train, y_train, X_test, y_test = data_loader(path)

    train_err = error_rate(final_theta, X_train, y_train)
    test_err = error_rate(final_theta, X_test, y_test)

    print(f"train error rate = {train_err}")
    print(f"test error rate  = {test_err}")

    with open(r'v_math329\HW1\results\q7_error_rates.txt', 'w') as file:
        file.write(f"train_error_rate={train_err}\n")
        file.write(f"test_error_rate={test_err}\n")

    return train_err, test_err
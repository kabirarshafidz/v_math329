import numpy as np
from helper import funcV, f_gradV
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat


def data_loader(path):
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    train, test = data['train'], data['test']

    X_train = train.X.T
    y_train = train.y.astype(float)

    X_test = test.X.T
    y_test = test.y.astype(float)

    return X_train, y_train, X_test, y_test


def GD(theta0, lr, max_time, tol, x_dat, y_label):
    theta = np.asarray(theta0, dtype='float')
    x_dat = np.asarray(x_dat, dtype='float')
    y_label = np.asarray(y_label, dtype='float')

    g0 = f_gradV(theta=theta, x_dat=x_dat, y_label=y_label)
    g0_norm = np.linalg.norm(g0)
    histry = {"theta": [], "f": [], "gnorm": [], "iteration": [], "reason": []}

    timeout = time.time() + 60 * max_time #inspo from the python time library documenation
    reason = f"Timeout at {max_time} minutes"
    iter = 0

    while time.time() <= timeout:
        grad_ = f_gradV(theta=theta, x_dat=x_dat, y_label=y_label)
        gn = np.linalg.norm(grad_)

        if gn <= g0_norm * tol:
            print(f"Gradient descent passed tolerance on iteration: {iter} with tol: {tol}")
            reason = f"Reached Tol: {tol}"
            break

        theta = theta - lr * grad_
        histry["gnorm"].append(gn)
        histry["theta"].append(theta.copy())
        histry["f"].append(funcV(theta, x_dat, y_label))
        histry["iteration"].append(int(iter))

        if iter % 100 == 0:
            print(f"Iteration: {iter}, New gnorm value: {gn}")
        iter += 1

    histry["reason"].append(reason)

    return theta, histry

def run_GD(lr=1e-3, tol=1e-3, path='../data/mnist_train_test.mat'):
    X_train, Y_train, X_test, Y_test = data_loader(path)
    rng = np.random.default_rng(seed=42)

    theta0 = rng.uniform(low=-0.01, high=0.01, size=785)

    final_theta, history = GD(theta0=theta0, lr=lr, max_time=3, x_dat=X_train, y_label=Y_train, tol=tol)

    plt.plot(history["iteration"], history["gnorm"])
    plt.xlabel("iteration k")
    plt.ylabel(r"$\|\nabla f_\lambda(\theta_k)\|$")
    plt.title("Gradient norm during GD")
    plt.savefig('../results/q4_GD_(optional).pdf')
    plt.close()

    stopping_reason = history['reason']

    n_iters = history['iteration'][-1]

    header_lines = (
        f"stopping_reason={stopping_reason}\n"
        f"total_iterations={n_iters}\n"
        "iteration,f,gnorm"
    )

    np.savetxt(
        r'../results/q4_history.csv',
        np.column_stack([history['iteration'], history['f'], history['gnorm']]),
        delimiter=',', header=header_lines, comments='# '
    )

    np.save('../results/final_theta.npy', final_theta)

    return final_theta, history

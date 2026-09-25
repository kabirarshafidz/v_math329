from Q2 import verification, time_comp
from Q3 import plot_
from Q4 import run_GD
from Q5 import plot_convergence
import numpy as np

DATA_PATH = r'data\mnist_train_test.mat'
STEP_SIZE = 1e-3
TOLERANCE = 1e-8


if __name__ == "__main__":
    verification()
    time_comp()
    plot_()
    final_theta, history = run_GD()

    np.save('results/final_theta.npy', final_theta)
    plot_convergence(path=DATA_PATH, history=history, lr=STEP_SIZE, tol=TOLERANCE)
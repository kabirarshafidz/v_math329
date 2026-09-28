from Q4 import run_GD
from Q5 import plot_convergence
from Q6 import q6_compare
from Q2_Arsha import run_q2
from Q3_Arsha import run_q3
import numpy as np

DATA_PATH = r'v_math329\HW1\data\mnist_train_test.mat'
STEP_SIZE = 1e-3
TOLERANCE = 1e-3


if __name__ == "__main__":
    run_q2()
    run_q3()
    final_theta, history = run_GD(lr=STEP_SIZE, tol=TOLERANCE)
    print(f"Convergence reason: {history['reason'][0]}")


    plot_convergence(path=DATA_PATH, history=history, lr=STEP_SIZE, tol=TOLERANCE)
    q6_compare(path=DATA_PATH)
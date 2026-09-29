from Q2_Arsha import run_q2
from v_math329.HW1.code.Q3 import run_q3
from Q4 import run_GD
from Q5 import q5_plot_convergence
from Q6 import q6_compare
from Q7 import q7_error_check

DATA_PATH = '../data/mnist_train_test.mat'
STEP_SIZE = 1e-3
TOLERANCE = 1e-3


def print_banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


if __name__ == "__main__":
    print_banner("QUESTION 2: Objective and Gradient Verification")
    run_q2()

    print_banner("QUESTION 3: Log-Log Gradient Check (Taylor Approximation)")
    run_q3()

    print_banner("QUESTION 4: Running Fixed-Step Gradient Descent")
    final_theta, history = run_GD(lr=STEP_SIZE, tol=TOLERANCE)
    print(f"Convergence reason: {history['reason'][0]}")

    print_banner("QUESTION 5: Plotting Convergence History")
    q5_plot_convergence(path=DATA_PATH, history=history, lr=STEP_SIZE, tol=TOLERANCE)
    print("Convergence plot generated and saved.")

    print_banner("QUESTION 6: Comparing Theory with Observed Convergence")
    q6_compare(path=DATA_PATH, history=history)

    print_banner("QUESTION 7: Evaluating Classifier Error Rates")
    q7_error_check(path=DATA_PATH, final_theta=final_theta)

    print("\n" + "=" * 60)
    print("  All tasks completed.")
    print("=" * 60 + "\n")
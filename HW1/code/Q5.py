from Q4 import run_GD, data_loader
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(path, history=None, lr=1e-3, tol=1e-8):
    if history == None:
        X_train, y_train, X_test, y_test = data_loader(path=path)

        _, history = run_GD(lr=lr, tol=tol, path=path)
    
    plt.plot(history['iteration'], np.log(history['gnorm']), np.log(history['f']))
    plt.savefig('results/q5_convergence.pdf')
    plt.show()

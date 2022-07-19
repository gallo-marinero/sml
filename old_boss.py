""" Using BOSS to solve the minimization problem
f(x) = sin(x) + 1.5*exp(-(x-4.3)**2) , 0 < x < 7
"""
import numpy as np
from boss.bo.bo_main import BOMain

def func(X):
    """ BOSS-compatible definition of the function. """
    x = X[0, 0]
    return np.sin(x) + 1.5*np.exp(-(x - 4.3)**2)

if __name__ == '__main__':
    bo = BOMain(
        func,
        np.array([[0., 7.]]),  # bounds
        yrange=[-1, 1],
        kernel='rbf',
        initpts=5,
        iterpts=15,
        verbosity=2
    )
    res = bo.run()
    print(res.xmin, res.fmin)

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF


def f(x):
    return (x-5) ** 2
# Training data x and y:
x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
y = f(x)
x = x.reshape(-1, 1)
# New input to predict:
x_new = np.array([15])
x_new = x_new.reshape(-1, 1)

gpr = gpr(kernel=None,random_state=0)
gpr.fit(x,y)
print(gpr.predict(x_new,return_std=True))

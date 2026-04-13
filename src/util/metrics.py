import numpy as np

def mae(X, Y):
    return np.mean(np.abs(X - Y))

def rmse(X, Y):
    return np.sqrt(np.mean((X - Y) ** 2))

def mrpd(X, Y):
    return np.mean(np.abs(X - Y) / (np.abs(X) + np.abs(Y)))*2

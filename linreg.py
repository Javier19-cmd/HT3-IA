from math import *

norm = lambda v: sqrt(sum(v**2))

def linear_regression(X, y, t, cost, grad, a=0.1, n=1000, on_step=None): # n = 10000
    
    costs = []

    for i in range(n):
        nabla = grad(X, y, t)
        
        t -= a * grad(X, y, t)

        costs.append(cost(X, y, t))

        if on_step:
            on_step(t)
    
    return t, costs
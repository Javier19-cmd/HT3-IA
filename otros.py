from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
import quad
import linreg

load_dotenv()

DATASET_SET_SIZE = int(os.environ["DATASET_SET_SIZE"])
DATASET_SPARCE_RATIO = int(os.environ["DATASET_SPARCE_RATIO"])
DATASET_X_LIM = int(os.environ["DATASET_X_LIM"])

# X es un dato dado.

def get_random_dataset(): # Obteniendo un dataset random.

    X = np.linspace(0, DATASET_X_LIM, DATASET_SET_SIZE).reshape((DATASET_SET_SIZE, 1))
    # Creando una X random
    Xr = np.hstack((
        np.ones((DATASET_SET_SIZE, 1)),
        X
    ))

    y = 3 + 2 * X + np.random.rand(DATASET_SET_SIZE, 1) * DATASET_SPARCE_RATIO

    return X, Xr, y

X, Xr, y = get_random_dataset()

to = np.random.rand(Xr.shape[1], 1) # Theta inicial.

print("to", to)
print(Xr)
print(Xr.shape, y.shape)

# plt.plot(X, y, "ro")
# #plt.plot(xm, ym)
# plt.show()

# def draw_each_step(t):
#     #DATASET_X_LIM = int(os.environ["DATASET_X_LIM"])

#     #plt.show()

#     xm = np.array([[0], [DATASET_X_LIM]])
#     xmr = np.hstack((
#         np.ones((2, 1)),
#         xm
#     ))
#     ym = xmr @ t

#     plt.plot(X, y, "ro")
#     plt.plot(xm, ym)
#     plt.show()

#q.cost, q.grad

tf, costs = linreg.linear_regression(
    Xr, 
    y, 
    to, 
    quad.cost, 
    quad.grad, 
    a=0.025, 
    n=20,
    #on_step=draw_each_step
    ) # Theta final.


print("Tf: ", tf)
xm = np.array([[0], [DATASET_X_LIM]])
xmr = np.hstack((
    np.ones((2, 1)),
    xm
))

ym = xmr @ tf
plt.plot(Xr[:, 1], y, "ro")
plt.plot(xm, ym)
plt.show()

# Costo.
plt.plot(costs)
plt.show()

# Gráfica de la recta.
#plt.plot(X, y, "ro")

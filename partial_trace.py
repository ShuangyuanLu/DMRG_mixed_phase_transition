import numpy as np


def partial_trace(rho, system, dims):   # dims = [dim_1, dim_2]
    rho = np.reshape(rho, dims + dims)
    if system == 1:
        rho = np.trace(rho, axis1=0, axis2=2)
    elif system == 2:
        rho = np.trace(rho, axis1=1, axis2=3)
    else:
        raise("Input Mistake.")

    return rho


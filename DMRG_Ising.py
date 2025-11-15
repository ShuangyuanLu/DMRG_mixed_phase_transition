from DMRG import DMRG
import numpy as np

class DMRG_Ising(DMRG):
    def __init__(self):
        parameters = {"J": 1, "h": 1, "L": 10, "cutoff_s": 10 ** (-8), "cutoff_n": 20, "n_sweep":10}
        hamiltonian_i = np.zeros([2, 2, 3, 3])
        hamiltonian_i[:, :, 0, 0] = np.eye(2)
        hamiltonian_i[:, :, 2, 2] = np.eye(2)
        hamiltonian_i[:, :, 0, 1] = - parameters["J"] * np.array([[1, 0], [0, -1]])
        hamiltonian_i[:, :, 1, 2] = np.array([[1, 0], [0, -1]])
        hamiltonian_i[:, :, 0, 2] = - parameters["h"] * np.array([[0, 1], [1, 0]])

        hamiltonian = [hamiltonian_i for _ in range(parameters["L"])]
        hamiltonian[0] = hamiltonian[0][:, :, 0: 1, :]
        hamiltonian[parameters["L"] - 1] = hamiltonian[parameters["L"] - 1][:, :, :,hamiltonian_i.shape[3] - 1: hamiltonian_i.shape[3]]

        super().__init__(hamiltonian, parameters)
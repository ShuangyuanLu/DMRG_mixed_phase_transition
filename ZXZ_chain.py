import numpy as np
from DMRG import DMRG


class ZXZ_chain(DMRG):
    def __init__(self, parameters):
        self.parameters = parameters
        # parameters = {'J' : 1, 'h' : 0.5, 'L': 10, "cutoff_s": 10 ** (-8), "cutoff_n": 50, "n_sweep": 100}
        hamiltonian = self.initialize_zxz_model()
        DMRG.__init__(self, hamiltonian, parameters)

    def initialize_zxz_model(self):
        id = np.eye(2)
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])

        hamiltonian_i = np.zeros([2, 2, 4, 4])
        hamiltonian_i[:, :, 0, 0] = id
        hamiltonian_i[:, :, 3, 3] = id
        hamiltonian_i[:, :, 0, 1] = - self.parameters['J'] * sz
        hamiltonian_i[:, :, 1, 2] = sx
        hamiltonian_i[:, :, 2, 3] = sz
        hamiltonian_i[:, :, 0, 3] = - self.parameters['h'] * sx

        hamiltonian = [hamiltonian_i for _ in range(self.parameters['L'])]
        hamiltonian[0] = hamiltonian[0][:, :, 0:1, :]
        hamiltonian[self.parameters['L']-1] = hamiltonian[self.parameters['L']-1][:, :, :, 3:4]

        return hamiltonian




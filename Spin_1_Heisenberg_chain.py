import numpy as np
from DMRG import DMRG


class Spin_1_Heisenberg_chain(DMRG):
    def __init__(self, parameters):
        self.parameters = parameters
        hamiltonian = self.initialize_zxz_model()
        DMRG.__init__(self, hamiltonian, parameters)

    def initialize_zxz_model(self):
        id = np.eye(3)
        sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
        sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
        sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        sp = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]) * np.sqrt(2)
        sm = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * np.sqrt(2)

        hamiltonian_i = np.zeros([3, 3, 5, 5])
        hamiltonian_i[:, :, 0, 0] = id
        hamiltonian_i[:, :, 4, 4] = id
        hamiltonian_i[:, :, 0, 1] = self.parameters['J'] * sp /2
        hamiltonian_i[:, :, 1, 4] = sm
        hamiltonian_i[:, :, 0, 2] = self.parameters['J'] * sm /2
        hamiltonian_i[:, :, 2, 4] = sp
        hamiltonian_i[:, :, 0, 3] = self.parameters['J'] * sz
        hamiltonian_i[:, :, 3, 4] = sz
        hamiltonian_i[:, :, 0, 4] = self.parameters['h'] * sz @ sz

        hamiltonian = [hamiltonian_i for _ in range(self.parameters['L'])]
        hamiltonian[0] = hamiltonian[0][:, :, 0:1, :]
        hamiltonian[self.parameters['L']-1] = hamiltonian[self.parameters['L']-1][:, :, :, 4:5]

        return hamiltonian




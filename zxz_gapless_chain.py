import numpy as np
from DMRG import DMRG


class ZXZ_gapless_chain(DMRG):
    def __init__(self, parameters):
        self.parameters = parameters
        # parameters = {'J' : 1, 'h' : 0.5, 'L': 10, "cutoff_s": 10 ** (-8), "cutoff_n": 50, "n_sweep": 100}
        hamiltonian = self.initialize_zxz_gapless_model()
        DMRG.__init__(self, hamiltonian, parameters)

    def initialize_zxz_gapless_model(self):
        id = np.eye(2)
        sx = np.array([[0, 1], [1, 0]])
        i_sy = np.array([[0, 1], [-1, 0]])
        sz = np.array([[1, 0], [0, -1]])

        hamiltonian_i = np.zeros([2, 2, 5, 5])
        hamiltonian_i[:, :, 0, 0] = id
        hamiltonian_i[:, :, 3, 3] = id
        hamiltonian_i[:, :, 0, 1] = - self.parameters['J'] * sz
        hamiltonian_i[:, :, 1, 2] = sx
        hamiltonian_i[:, :, 2, 3] = sz
        hamiltonian_i[:, :, 0, 3] = - self.parameters['h'] * sx

        hamiltonian_i_even = hamiltonian_i.copy()
        hamiltonian_i_odd = hamiltonian_i.copy()

        hamiltonian_i_odd[:, :, 0, 4] = - self.parameters['K'] * i_sy
        hamiltonian_i_odd[:, :, 4, 3] = (-i_sy)

        hamiltonian_i_even[:, :, 4, 4] = sx

        hamiltonian = []
        for site in range(self.parameters['L']):
            if site % 2 == 0:
                hamiltonian.append(hamiltonian_i_even)
            else:
                hamiltonian.append(hamiltonian_i_odd)
        hamiltonian[0] = hamiltonian[0][:, :, 0:1, :]
        hamiltonian[self.parameters['L']-1] = hamiltonian[self.parameters['L']-1][:, :, :, 3:4]

        return hamiltonian




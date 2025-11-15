import numpy as np
from scipy.sparse.linalg import LinearOperator
from functools import partial
import scipy
from DMRG_NESS import DMRG_NESS
import math


class NESS_Heisenberg:
    def __init__(self, parameters):
        # parameters = {'h': 1, 'J': 0.3, 'J1': 0.5, 'L': 8, "cutoff_s": 10 ** (-8), "cutoff_n": 50, "n_sweep": 100}
        self.parameters = parameters
        self.L = self.parameters['L']
        self.evolution_matrix_J = None

        self.generate_matrix()

    def generate_matrix(self):
        id = np.eye(2)
        id_4 = np.eye(4)
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])

        H_heisenberg = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
        state_singlet = np.array([0, 1, 1, 0]) / math.sqrt(2)
        rho_env_J = np.outer(state_singlet, np.conjugate(state_singlet))
        # coupling env and sys
        H_heisenberg_1 = np.kron(np.kron(sx, id), np.kron(sx, id)) + np.kron(np.kron(sy, id), np.kron(sy, id)) + np.kron(np.kron(sz, id), np.kron(sz, id))
        H_heisenberg_2 = np.kron(np.kron(id, sx), np.kron(id, sx)) + np.kron(np.kron(id, sy), np.kron(id, sy)) + np.kron(np.kron(id, sz), np.kron(id, sz))

        J = self.parameters['J']
        J1 = self.parameters['J1']

        H_J = J * np.kron(id_4, H_heisenberg) + J1 * (H_heisenberg_1 + H_heisenberg_2)
        U_J = scipy.linalg.expm(1j * H_J)
        evolution_operator_J = LinearOperator((4 ** 2, 4 ** 2), matvec=partial(DMRG_NESS.evolution, dim=4, U=U_J, rho_env=rho_env_J))

        self.evolution_matrix_J = DMRG_NESS.linear_operator_to_matrix(evolution_operator_J, 4 ** 2).transpose()



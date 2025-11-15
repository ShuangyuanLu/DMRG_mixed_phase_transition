import numpy as np
from scipy.sparse.linalg import LinearOperator
from functools import partial
import scipy
from DMRG_NESS import DMRG_NESS


class NESS_Ising:
    def __init__(self, parameters):
        self.parameters = parameters
        self.L = self.parameters['L']
        self.evolution_matrix_J = None
        self.evolution_matrix_h = None

        self.generate_matrix()

    def generate_matrix(self):
        id = np.eye(2)
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])
        sp = np.array([[0, 1], [0, 0]])
        sm = np.array([[0, 0], [1, 0]])
        Pp = np.array([[1, 0], [0, 0]])
        Pm = np.array([[0, 0], [0, 1]])

        rho_env_J = (np.kron(id, id) + np.kron(sz, sz)) / 4
        H_J = np.kron(np.kron(Pp, sm), np.kron(Pp, sp)) + np.kron(np.kron(Pp, sp), np.kron(Pp, sm)) + np.kron(
            np.kron(sm, Pp), np.kron(sp, Pp)) + np.kron(np.kron(sp, Pp), np.kron(sm, Pp)) \
              + np.kron(np.kron(Pm, sm), np.kron(Pm, sp)) + np.kron(np.kron(Pm, sp), np.kron(Pm, sm)) + np.kron(
            np.kron(sm, Pm), np.kron(sp, Pm)) + np.kron(np.kron(sp, Pm), np.kron(sm, Pm))
        U_J = scipy.linalg.expm(1j * self.parameters['J'] * H_J)

        rho_env_h = (id + sx) / 2
        H_h = np.kron(sz, sz) + np.kron(sy, sy)
        U_h = scipy.linalg.expm(1j * self.parameters['h'] * H_h)

        evolution_operator_J = LinearOperator((4 ** 2, 4 ** 2),
                                              matvec=partial(DMRG_NESS.evolution, dim=4, U=U_J, rho_env=rho_env_J))
        evolution_operator_h = LinearOperator((2 ** 2, 2 ** 2),
                                              matvec=partial(DMRG_NESS.evolution, dim=2, U=U_h, rho_env=rho_env_h))

        self.evolution_matrix_J = DMRG_NESS.linear_operator_to_matrix(evolution_operator_J, 4 ** 2).transpose()
        self.evolution_matrix_h = DMRG_NESS.linear_operator_to_matrix(evolution_operator_h, 2 ** 2).transpose()


from DMRG import DMRG
import numpy as np
from partial_trace import partial_trace
from MPS_DM_basic import MPS_DM_basic
import math
import cmath


class DMRG_NESS(DMRG, MPS_DM_basic):
    def __init__(self, evolution, parameters):
        DMRG.__init__(self, evolution, parameters)
        self.eigenstate_mode = 'SM'

    def run(self):
        DMRG.run(self)
        self.normalize()

    def measure_site(self, site, operator):
        return MPS_DM_basic.measure_site(self, site, operator)

    def trace(self):
        return MPS_DM_basic.trace(self)

    def normalize(self):
        MPS_DM_basic.normalize(self)

    @staticmethod
    def evolution(rho_sys_input, dim, U, rho_env):
        rho_sys_input = np.reshape(rho_sys_input, [dim, dim])
        rho_sys_output = partial_trace(np.matmul(np.matmul(U, np.kron(rho_env, rho_sys_input)), (U.conj().transpose())), 1, [dim, dim])
        rho_sys_output = np.reshape(rho_sys_output, [dim ** 2])
        return rho_sys_output

    @staticmethod
    def linear_operator_to_matrix(operator, n):
        basis_vectors = np.eye(n)  # Standard basis vectors (identity matrix)
        return np.column_stack([operator(basis_vectors[:, i]) for i in range(n)])
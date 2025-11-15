from DMRG_NESS import DMRG_NESS
from NESS_Ising import NESS_Ising
import numpy as np
from MPS_DM_basic import MPS_DM_basic


class DMRG_NESS_Ising(DMRG_NESS, NESS_Ising, MPS_DM_basic):
    def __init__(self, parameters):
        NESS_Ising.__init__(self, parameters)
        evolution_list = self.generate_evolution()
        DMRG_NESS.__init__(self, evolution=evolution_list, parameters=self.parameters)

    def generate_evolution(self):
        evolution_matrix_ij = self.evolution_matrix_J.reshape((2, 2, 2, 2, 2, 2, 2, 2))
        evolution_matrix_ij = evolution_matrix_ij.transpose((0, 2, 4, 6, 1, 3, 5, 7))
        evolution_matrix_ij = evolution_matrix_ij.reshape((4 ** 2, 4 ** 2))
        U, S, V = np.linalg.svd(evolution_matrix_ij)

        D = np.sum(S > 10 ** (-10))
        evolution_left = U * S
        evolution_left = evolution_left[:, 0: D].reshape((4, 4, D))
        evolution_right = V[0: D, :].reshape((D, 4, 4))

        evolution_odd = np.einsum('dab,bce->acde', evolution_right, evolution_left)
        evolution_even = np.einsum('abe,dbc->acde', evolution_left, evolution_right)
        evolution_left = evolution_left[:, :, np.newaxis, :]
        evolution_right = evolution_right.transpose((1, 2, 0))[:, :, :, np.newaxis]

        evolution_list = [evolution_left]
        for site in range(1, self.L - 1):
            if site % 2 == 0:
                evolution_list.append(evolution_even)
            else:
                evolution_list.append(evolution_odd)
        evolution_list.append(evolution_right)

        for site in range(self.L):
            evolution_site = evolution_list[site]
            evolution_site = np.einsum('abcd,be->aecd', evolution_site, self.evolution_matrix_h)

            shape_site = evolution_site.shape
            new_shape_site = (shape_site[0], shape_site[1], shape_site[2] + 1 * (site != 0), shape_site[3] + 1 * (site != self.L - 1))
            evolution_site_1 = np.zeros(new_shape_site, dtype=np.complex128)
            evolution_site_1[:, :, :shape_site[2], :shape_site[3]] = evolution_site
            evolution_site_1[:, :, new_shape_site[2] - 1, new_shape_site[3] - 1] = np.eye(4) * (1 - 2 * (site == 0))
            evolution_site_1 = np.einsum('acbd,fceg->afbedg', evolution_site_1, evolution_site_1.conjugate()).reshape((new_shape_site[0], new_shape_site[1], new_shape_site[2] ** 2, new_shape_site[3] ** 2))
            evolution_list[site] = evolution_site_1

        return evolution_list

    def measure_site(self, site, operator):
        return MPS_DM_basic.measure_site(self, site, operator)

    def trace(self):
        return MPS_DM_basic.trace(self)

    def normalize(self):
        MPS_DM_basic.normalize(self)




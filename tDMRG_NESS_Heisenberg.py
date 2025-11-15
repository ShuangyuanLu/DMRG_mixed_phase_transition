import numpy as np
from NESS_Heisenberg import NESS_Heisenberg
from tDMRG import tDMRG
from tDMRG_NESS import tDMRG_NESS
from MPS_DM_basic import MPS_DM_basic


class tDMRG_NESS_Heisenberg(tDMRG_NESS, NESS_Heisenberg):
    def __init__(self, parameters):
        NESS_Heisenberg.__init__(self, parameters)
        evolution_list = self.generate_evolution()
        tDMRG_NESS.__init__(self, evolution_list, parameters)

    def generate_evolution(self):
        evolution_matrix_ij = self.evolution_matrix_J.reshape((2, 2, 2, 2, 2, 2, 2, 2))
        evolution_matrix_ij = evolution_matrix_ij.transpose((0, 2, 4, 6, 1, 3, 5, 7))
        evolution_matrix_ij = evolution_matrix_ij.reshape((4, 4, 4, 4))
        # transpose is in the class NESS_Heisenberg

        evolution_list = [[], []]
        for site in range(self.L - 1):
            if site % 2 == 0:
                evolution_list[1].append(evolution_matrix_ij)
                evolution_list[0].append(None)
            else:
                evolution_list[1].append(None)
                evolution_list[0].append(evolution_matrix_ij)

        return evolution_list


import numpy as np
from MPS_basic import MPS_basic
import math


class tDMRG(MPS_basic):
    def __init__(self, evolution, parameters):
        super().__init__(parameters)
        self.n_trotter = 2
        self.evolution = evolution
        self.d = next((x for x in evolution[0] if x is not None), None).shape[0]

        self.result = {}

        state_site = np.zeros([self.d, 1, 1])
        state_site[0, 0, 0] = 1
        self.state = [state_site for _ in range(self.L)]

    def sweep(self):
        for site in range(0, self.L - 1):
            state_2 = np.einsum('bae,ced->abcd', self.state[site], self.state[site + 1])
            if self.evolution[0][site] is not None:
                state_2 = np.einsum('abcd,becf->aefd', state_2, self.evolution[0][site])
                state_2 = state_2 #/ math.sqrt(np.abs(np.einsum('abcd,abcd->', state_2, state_2.conj())))
            self.svd(site, 'left_to_right', state_2)
        for site in range(self.L - 2, -1, -1):
            state_2 = np.einsum('bae,ced->abcd', self.state[site], self.state[site + 1])
            if self.evolution[1][site] is not None:
                state_2 = np.einsum('abcd,becf->aefd', state_2, self.evolution[1][site])
                state_2 = state_2 #/ math.sqrt(np.abs(np.einsum('abcd,abcd->', state_2, state_2.conj())))
            self.svd(site, 'right_to_left', state_2)

    def run(self):
        for i_sweep in range(self.n_sweep):
            self.sweep()

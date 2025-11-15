from tDMRG import tDMRG
from MPS_DM_basic import MPS_DM_basic
import numpy as np
from partial_trace import partial_trace


class tDMRG_NESS(tDMRG, MPS_DM_basic):
    def __init__(self, evolution, parameters):
        tDMRG.__init__(self, evolution, parameters)

    def run(self):
        for i_sweep in range(self.n_sweep):
            self.sweep()
        print("trace:", self.trace())
        self.normalize()

    def measure_site(self, site, operator):
        return MPS_DM_basic.measure_site(self, site, operator)

    def measure_2_sites(self, site_1, site_2, operator):
        return MPS_DM_basic.measure_2_sites(self, site_1, site_2, operator)

    def measure_time_corr(self, site_1, site_2, operator):
        self.state[site_1] = np.einsum("abc, ad -> dbc", self.state[site_1], operator)
        for _ in range(10):
            self.sweep()
        self.state[site_2] = np.einsum("abc, ad -> dbc", self.state[site_2], operator)
        corr = self.trace()

        return corr

    def trace(self):
        return MPS_DM_basic.trace(self)

    def normalize(self):
        MPS_DM_basic.normalize(self)
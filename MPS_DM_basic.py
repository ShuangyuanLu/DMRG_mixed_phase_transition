import numpy as np
from MPS_basic import MPS_basic
import math
import cmath

class MPS_DM_basic(MPS_basic):
    def __init__(self, parameters):
        super().__init__(parameters)

    def measure_site(self, site, operator):
        operator = operator.reshape(-1)
        trace_operator = np.eye(math.isqrt(self.d)).reshape(-1)
        right_operator = np.array([1])
        for i in range(self.L - 1, -1, -1):
            if i == site:
                site_trace = np.einsum('abc,a->bc', self.state[i], operator)
            else:
                site_trace = np.einsum('abc,a->bc', self.state[i], trace_operator)
            right_operator = np.einsum('ab,b->a', site_trace, right_operator)
        return right_operator[0]

    def measure_2_sites(self, site_1, site_2, operator):
        operator = operator.reshape(-1)
        trace_operator = np.eye(math.isqrt(self.d)).reshape(-1)
        right_operator = np.array([1])
        for i in range(self.L - 1, -1, -1):
            if i == site_1:
                site_trace = np.einsum('abc,a->bc', self.state[i], operator)
            elif i == site_2:
                site_trace = np.einsum('abc,a->bc', self.state[i], operator)
            else:
                site_trace = np.einsum('abc,a->bc', self.state[i], trace_operator)
            right_operator = np.einsum('ab,b->a', site_trace, right_operator)
        return right_operator[0]

    def trace(self):
        trace_operator = np.eye(math.isqrt(self.d)).reshape(-1)
        right_operator = np.array([1])
        for site in range(self.L - 1, -1, -1):
            site_trace = np.einsum('abc,a->bc', self.state[site], trace_operator)
            right_operator = np.einsum('ab,b->a', site_trace, right_operator)
        return right_operator[0]

    def normalize(self):
        trace_state = self.trace()
        trace_site = cmath.exp(cmath.log(trace_state) / self.L)
        for i in range(self.L):
            self.state[i] = self.state[i] / trace_site

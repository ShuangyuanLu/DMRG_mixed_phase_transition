import numpy as np


class MPS_basic:
    def __init__(self, parameters):
        self.parameters = parameters
        self.cutoff_n = parameters['cutoff_n']
        self.cutoff_s = parameters['cutoff_s']
        self.L = parameters['L']  # system size
        self.n_sweep = parameters['n_sweep']  # max sweep n
        self.state = None

    def svd(self, site, direction, state_2):
        shape = state_2.shape
        state_2 = state_2.reshape((shape[0] * shape[1], shape[2] * shape[3]))
        U, S, V = np.linalg.svd(state_2, full_matrices=False)
        new_D = min(self.cutoff_n, np.sum(S > self.cutoff_s * S[0]))
        #print("S:", S)
        U = U[:, :new_D]
        S = S[:new_D]
        V = V[:new_D, :]
        if direction == 'left_to_right':
            state_site = U
            state_site_plus_1 = S[:, np.newaxis] * V
        elif direction == 'right_to_left':
            state_site = U * S
            state_site_plus_1 = V
        else:
            raise Exception('Direction must be either "left_to_right" or "right_to_left"')
        #print("1:", site,  self.state[site])
        #print("2:", site + 1, self.state[site + 1])
        self.state[site] = np.transpose(state_site.reshape([shape[0], shape[1], new_D]), [1, 0, 2])
        self.state[site + 1] = np.transpose(state_site_plus_1.reshape([new_D, shape[2], shape[3]]), [1, 0, 2])

    def norm(self):
        operator_right = np.eye(1)
        for i in range(self.L-1, -1, -1):
            operator_right = np.einsum('cf,bac->abf', operator_right, self.state[i])
            operator_right = np.einsum('aef,edf->ad', operator_right, self.state[i].conjugate())
        return operator_right[0, 0]

    def measure_site(self, site, operator):
        operator_right = np.eye(self.state[site].shape[2])
        for i in range(site, -1, -1):
            operator_right = np.einsum('cf,bac->abf', operator_right, self.state[i])
            if i == site:
                operator_right = np.einsum('abf,be->aef', operator_right, operator)
            operator_right = np.einsum('aef,edf->ad', operator_right, self.state[i].conjugate())
        return operator_right[0, 0]

    def measure_site_list(self, site_list, operator_list):
        operator_right = np.eye(self.state[self.L-1].shape[2])
        for i in range(self.L - 1, -1, -1):
            operator_right = np.einsum('cf,bac->abf', operator_right, self.state[i])
            if i in site_list:
                operator_right = np.einsum('abf,be->aef', operator_right, operator_list[site_list.index(i)])
            operator_right = np.einsum('aef,edf->ad', operator_right, self.state[i].conjugate())
        return operator_right[0, 0]

    '''
    def measure_2_sites(self, site_1, site_2, operator):
        if site_1 >= site_2:
            raise Exception('Site 1 must be smaller than Site 2')
        operator_right = np.eye(self.state[site_2].shape[2])
        for i in range(site_2, -1, -1):
            operator_right = np.einsum('cf,bac->abf', operator_right, self.state[i])
            if i == site_2:
                operator_right = np.einsum('abf,be->aef', operator_right, operator)
            if i == site_1:
                operator_right = np.einsum('abf,be->aef', operator_right, operator)
            operator_right = np.einsum('aef,edf->ad', operator_right, self.state[i].conjugate())
        return operator_right[0, 0]
    '''



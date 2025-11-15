import numpy as np
from functools import partial
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from MPS_basic import MPS_basic
import math
import time
from scipy.sparse.linalg import eigsh

# finite dmrg
class DMRG(MPS_basic):
    def __init__(self, hamiltonian, parameters):
        MPS_basic.__init__(self, parameters)
        self.eigenstate_mode = 'SR'
        self.hamiltonian = hamiltonian  # tensor
        self.d = hamiltonian[0].shape[0]   # physical link dim
        self.DH = hamiltonian[0].shape[3]

        self.eigenvalue_list = []
        self.result = {}

        state_site = np.zeros([self.d, 1, 1])
        state_site[0, 0, 0] = math.sqrt(3)/2
        state_site[1, 0, 0] = 1 /2
        self.state = [state_site for _ in range(self.L)]
        self.left = [np.ones([1, 1, 1]) for _ in range(self.L)]
        self.right = [np.ones([1, 1, 1]) for _ in range(self.L)]
        self.prepare_env()

    def prepare_env(self):
        for site in range(0, self.L - 1):
            self.update_left(site)
        for site in range(self.L - 1, 0, -1):
            self.update_right(site)

    @staticmethod
    def map(state_2, left, right, hamiltonian_1, hamiltonian_2, shape):
        state_2 = state_2.reshape(shape)
        left_state = np.einsum('aej,abcd->bcdej', left, state_2)
        left_state_hamiltonian_1 = np.einsum('bcdej,bfeg->cdfgj', left_state, hamiltonian_1)
        left_state_hamiltonian = np.einsum('cdfgj,chgi->dfhij', left_state_hamiltonian_1, hamiltonian_2)
        final_state = np.einsum('dfhij,dik->jfhk', left_state_hamiltonian, right)
        return final_state.reshape(-1)

    '''
    def update_state(self, site, direction, update_or_not):
        state_2 = np.einsum('bae,ced->abcd', self.state[site], self.state[site + 1])
        shape = state_2.shape

        if update_or_not:
            hamiltonian_site_function = partial(DMRG.map, left=self.left[site], right=self.right[site + 1], hamiltonian_1=self.hamiltonian[site], hamiltonian_2=self.hamiltonian[site + 1], shape=shape)
            # noinspection PyArgumentList
            hamiltonian_site = LinearOperator((state_2.size, state_2.size), matvec=hamiltonian_site_function)
            # noinspection PyTypeChecker
            eigenvalue, state_2 = eigs(hamiltonian_site, k=1, which=self.eigenstate_mode, v0=state_2.reshape(-1))
            self.eigenvalue_list.append(eigenvalue[0])
            state_2 = state_2.reshape(shape)

        self.svd(site, direction, state_2)
    '''

    def update_state(self, site, direction, update_or_not):
        state_2 = np.einsum('bae,ced->abcd', self.state[site], self.state[site + 1])
        shape = state_2.shape

        if update_or_not:
            x0 = state_2.reshape(-1)
            x0 = x0 / np.linalg.norm(x0)
            N = x0.size
            dtype = x0.dtype

            left = self.left[site]
            right = self.right[site + 1]
            h1 = self.hamiltonian[site]
            h2 = self.hamiltonian[site + 1]
            shape = shape  # (Dl, d, d, Dr) or whatever yours is

            def matvec(x_flat):
                # avoid new big allocations; reshape is a view
                x = x_flat.reshape(shape)
                y = DMRG.map(x, left, right, h1, h2, shape)  # ideally writes into a preallocated buffer
                return y.reshape(N)

            H = LinearOperator((N, N), matvec=matvec, dtype=dtype)

            # Lanczos tuned for k=1
            evals, evecs = eigsh(H, k=1, which='SA', v0=x0, ncv=32, tol=1e-8, maxiter=1000)
            e0 = evals[0]
            state_2 = evecs[:, 0].reshape(shape)
            self.eigenvalue_list.append(e0)

        self.svd(site, direction, state_2)

    def update_left(self, site):
        new_left = np.einsum('adg,bac->bcdg', self.left[site], self.state[site])
        new_left = np.einsum('bcdg,bedf->cefg', new_left, self.hamiltonian[site])
        new_left = np.einsum('cefg,egh->cfh', new_left, self.state[site].conjugate())
        self.left[site + 1] = new_left

    def update_right(self, site):
        new_right = np.einsum('cfh,bac->abfh', self.right[site], self.state[site])
        new_right = np.einsum('abfh,bedf->adeh', new_right, self.hamiltonian[site])
        new_right = np.einsum('adeh,egh->adg', new_right, self.state[site].conjugate())
        self.right[site - 1] = new_right

    def sweep(self, i_sweep):
        #if i_sweep % 10 == 0:
        #    print(i_sweep)
        for site in range(self.L - 1):
            self.update_state(site, direction='left_to_right', update_or_not=(site != 0 or i_sweep == 0))
            self.update_left(site)
        for site in range(self.L - 2, -1, -1):
            self.update_state(site, direction='right_to_left', update_or_not=(site != self.L - 2))
            self.update_right(site + 1)

    def run(self):
        for i_sweep in range(self.n_sweep):
            self.sweep(i_sweep)

        self.result["energy"] = self.eigenvalue_list[-1]
        self.result["uncertainty"] = self.uncertainty()

    def uncertainty(self):
        magnitude = np.einsum('abc,abc->', self.state[0], self.state[0].conjugate())

        H_exp = np.einsum('cfh,bac->abfh', self.right[0], self.state[0])
        H_exp = np.einsum('abfh,bedf->adeh', H_exp, self.hamiltonian[0])
        H_exp = np.einsum('adeh,egh->adg', H_exp, self.state[0].conjugate())
        H_exp = np.einsum('adg,adg->', H_exp, self.left[0]) / magnitude

        right_2 = np.ones([1, 1, 1, 1])
        for site in range(self.L - 1, -1, -1):
            right_2 = np.einsum('cfik,bac->abfik', right_2, self.state[site])
            right_2 = np.einsum('abfik,bedf->adeik', right_2, self.hamiltonian[site])
            right_2 = np.einsum('adeik,ehgi->adghk', right_2, self.hamiltonian[site])
            right_2 = np.einsum('adghk,hjk->adgj', right_2, self.state[site].conjugate())
        H_2_exp = np.einsum('adgj,adgj->', right_2, np.ones([1, 1, 1, 1])) / magnitude

        return np.sqrt(H_2_exp - H_exp ** 2)

    def print_hamiltonian(self, i):
        hamiltonian_matrix = np.transpose(self.hamiltonian[i], [2, 0, 3, 1]).reshape(self.hamiltonian[i].shape[2] * self.hamiltonian[i].shape[0], self.hamiltonian[i].shape[3] * self.hamiltonian[i].shape[1])
        print(hamiltonian_matrix)

    def check_shape(self):
        for i in range(self.L):
            print(self.left[i].shape, self.right[i].shape, self.state[i].shape)


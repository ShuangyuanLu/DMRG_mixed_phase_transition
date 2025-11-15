import numpy as np
import shelve
from ZXZ_chain import ZXZ_chain
import random
import matplotlib.pyplot as plt
import time
import math


class CMI_MC:
    def __init__(self, file_name, model_type="zxz"):
        self.id = np.array([[1, 0], [0, 1]])
        self.sx = np.array([[0, 1], [1, 0]])
        self.sz = np.array([[1, 0], [0, -1]])

        with shelve.open(file_name) as db:
            state = db["state"]
            parameters = db["parameters"]

        for i in range(len(state)):
            state[i] = state[i].transpose(1, 0, 2)

        X_even_matrix = np.zeros([4, 4, 2, 2])
        X_odd_matrix = np.zeros([4, 4, 2, 2])
        X_even_matrix[0, 0, :, :] = np.eye(2)
        X_even_matrix[1, 1, :, :] = self.sx
        X_even_matrix[2, 2, :, :] = np.eye(2)
        X_even_matrix[3, 3, :, :] = self.sx
        X_odd_matrix[0, 0, :, :] = np.eye(2)
        X_odd_matrix[1, 1, :, :] = np.eye(2)
        X_odd_matrix[2, 2, :, :] = self.sx
        X_odd_matrix[3, 3, :, :] = self.sx
        boundary = np.ones([1, 4])
        for i in range(len(state)):
            state_i = state[i]
            if i % 2 == 0:
                X_matrix = X_even_matrix
            else:
                X_matrix = X_odd_matrix

            if i == 0:
                X_matrix_1 = np.einsum("abxy,ca->cbxy", X_matrix, boundary)/2
            elif i == len(state) - 1:
                X_matrix_1 = np.einsum("abxy,cb->acxy", X_matrix, boundary)/2
            else:
                X_matrix_1 = X_matrix

            state_i = np.einsum("xab,cdxy->yacbd", state_i, X_matrix_1)
            D0, D1, D2, D3, D4 = state_i.shape
            state_i = state_i.reshape(D0, D1 * D2, D3 * D4)
            state[i] = state_i

        zx_rotation = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        for i in range(len(state)):
            if i % 2 == 1:
                state[i] = np.einsum("iab,ij->jab", state[i], zx_rotation)

        my_mps = ZXZ_chain({'J': 1, 'h': parameters['h'], 'L': len(state),
                            "cutoff_s": None, "cutoff_n": None, "n_sweep": None})
        my_mps.state = state

        norm = np.sqrt(my_mps.norm())
        print("norm:", norm)
        state[0] = state[0] / norm

        random.seed(0)

        self.state = state
        self.L = len(state)
        self.x_0 = self.L // 8 * 2
        self.x_1 = self.L // 8 * 6

        self.projection_sites = [2*i+1 for i in range(self.L // 2)]
        self.n_projection = len(self.projection_sites)

        self.operator_right = None
        self.operator_left = None
        self.operator_left_record = None
        self.operator_right_record = None

        #self.mc_state = [random.randint(0, 1) for _ in range(self.L // 2)]
        self.mc_state = [0 for _ in range(self.L // 2)] # to keep even charge
        self.weight = self.contraction_after_projection()
        self.operator_left = self.operator_left_record
        self.operator_right = self.operator_right_record

        self.n_mc = 1
        self.n_measure = 1
        self.cmi = []
        self.negativity = []
        self.mid = self.L // 2

    def fast_calculation_negativity_cmi(self):
        self.cmi = self.contraction_with_two_sites_left()
        self.negativity = self.middle_canonical_form()
        print("cmi:", self.cmi)
        print("negativity:", self.negativity)

    # def mc_run(self):
    #     start = time.time()
    #     for i_mc in range(self.n_mc):
    #         self.sweep()
    #         if i_mc % self.n_measure == self.n_measure - 1:
    #             self.measure()
    #     end = time.time()
    #     print("mc_time:", end - start)
    #
    #     plt.plot(self.cmi)
    #     plt.plot(self.negativity)
    #     plt.savefig('cmi.png')
    #
    #     cmi_useful = np.array(self.cmi[self.n_mc // (self.n_measure * 2):])
    #     negativity_useful = np.array(self.negativity[self.n_mc // (self.n_measure * 2):])
    #     print("cmi_ave:", np.mean(cmi_useful), np.std(cmi_useful) / np.sqrt(cmi_useful.shape[0]))
    #     print("negativity_ave:", np.mean(negativity_useful), np.std(negativity_useful) / np.sqrt(negativity_useful.shape[0]))
    #     print(self.negativity)
    #
    # def measure(self):
    #     self.cmi.append(self.contraction_with_two_sites_left())
    #     self.negativity.append(self.middle_canonical_form())
    #
    # def sweep(self):
    #     for site in range(self.n_projection):
    #         self.update_two_sites(site)
    #
    # def update_site(self, site):
    #     self.mc_state[site] = 1 - self.mc_state[site]
    #     new_weight = self.contraction_after_projection()
    #     if random.random() < (new_weight / self.weight):
    #         self.weight = new_weight
    #         self.operator_left = self.operator_left_record
    #         self.operator_right = self.operator_right_record
    #     else:
    #         self.mc_state[site] = 1 - self.mc_state[site]
    #
    # def update_two_sites(self, site):
    #     print(self.mc_state)
    #     the_other_site = random.randint(0, self.n_projection - 1)
    #     self.mc_state[site] = 1 - self.mc_state[site]
    #     self.mc_state[the_other_site] = 1 - self.mc_state[the_other_site]
    #     new_weight = self.contraction_after_projection()
    #     if random.random() < (new_weight / self.weight):
    #         self.weight = new_weight
    #         self.operator_left = self.operator_left_record
    #         self.operator_right = self.operator_right_record
    #     else:
    #         self.mc_state[site] = 1 - self.mc_state[site]
    #         self.mc_state[the_other_site] = 1 - self.mc_state[the_other_site]

    def contraction_after_projection(self):
        operator_right = np.ones([1, 1])
        operator_left = np.ones([1, 1])
        projection_sites = self.projection_sites.copy()
        projection_spins = self.mc_state.copy()

        for site in range(self.x_0):
            if projection_sites and site == projection_sites[0]:
                del projection_sites[0]
                spin_site = projection_spins.pop(0)
                state_site = self.state[site][spin_site, :, :]
                operator_left = np.einsum("cd,db->cb", operator_left, state_site)
                operator_left = np.einsum("cb,ca->ab", operator_left, state_site.conj()) * 2
            else:
                operator_left = np.einsum("cd,xdb->xcb", operator_left, self.state[site])
                operator_left = np.einsum("xcb,xca->ab", operator_left, self.state[site].conj())
        self.operator_left_record = operator_left

        for site in range(self.L-1, self.x_0 - 1, -1):
            if projection_sites and site == projection_sites[-1]:
                del projection_sites[-1]
                spin_site = projection_spins.pop()
                state_site = self.state[site][spin_site, :, :]
                operator_right = np.einsum("ab,db->ad", operator_right, state_site)
                operator_right = np.einsum("ad,ca->cd", operator_right, state_site.conj()) * 2
            else:
                operator_right = np.einsum("ab,xdb->xad", operator_right, self.state[site])
                operator_right = np.einsum("xad,xca->cd", operator_right, self.state[site].conj())

            if site == self.x_1 + 1:
                self.operator_right_record = operator_right

        result = np.einsum("ab,ab->", operator_left, operator_right)
        return result

    def contraction_with_two_sites_left(self):
        '''
        ---- ----------- ----
            |           |
            y           z

            x           w
            |           |
        ---- ----------- ----
        '''
        projection_sites = [self.projection_sites[i] for i in range(len(self.projection_sites)) if self.x_0 < self.projection_sites[i] < self.x_1]
        projection_spins = [self.mc_state[i] for i in range(len(self.mc_state)) if self.x_0 < self.projection_sites[i] < self.x_1]

        operator_left = np.einsum("cd,xdb->xcb", self.operator_left, self.state[self.x_0])
        operator_left = np.einsum("xcb,yca->xyab", operator_left, self.state[self.x_0].conj())

        operator_right = np.einsum("ab,wdb->wad", self.operator_right, self.state[self.x_1])
        operator_right = np.einsum("wad,zca->wzcd", operator_right, self.state[self.x_1].conj())

        for site in range(self.x_1 - 1, self.x_0, -1):
            if projection_sites and site == projection_sites[-1]:
                del projection_sites[-1]
                spin_site = projection_spins.pop()
                state_site = self.state[site][spin_site, :, :]
                operator_right = np.einsum("wzab,db->wzad", operator_right, state_site)
                operator_right = np.einsum("wzad,ca->wzcd", operator_right, state_site.conj()) * 2
            else:
                operator_right = np.einsum("wzab,xdb->wzxad", operator_right, self.state[site])
                operator_right = np.einsum("wzxad,xca->wzcd", operator_right, self.state[site].conj())

        two_site_density_matrix = np.einsum("xyab,wzab->xywz", operator_left, operator_right)
        two_site_density_matrix = two_site_density_matrix / self.weight

        rho_2 = two_site_density_matrix.transpose(0, 2, 1, 3).reshape(4, 4)
        rho_1_A = np.einsum("xyww->xy", two_site_density_matrix)
        rho_1_B = np.einsum("xxwz->wz", two_site_density_matrix)
        cmi = CMI_MC.entropy(rho_1_A) + CMI_MC.entropy(rho_1_B) - CMI_MC.entropy(rho_2)
        #cmi = cmi * self.weight / 2 ** self.n_projection
        return cmi

    def middle_canonical_form(self):
        projection_sites = self.projection_sites.copy()
        projection_spins = self.mc_state.copy()

        left_matrix = np.ones([1, 1])
        for site in range(self.mid):
            if projection_sites and site == projection_sites[0]:
                del projection_sites[0]
                spin_site = projection_spins.pop(0)
                state_site = self.state[site][spin_site, :, :]
                left_matrix = left_matrix @ state_site * np.sqrt(2)
            else:
                state_site = np.einsum("xab,ca->xcb", self.state[site], left_matrix)
                D0, D1, D2 = state_site.shape
                state_site = state_site.reshape(D0*D1, D2)
                U, S, Vh = np.linalg.svd(state_site, full_matrices=False)
                left_matrix = np.diag(S) @ Vh

        right_matrix = np.ones([1, 1])
        for site in range(self.L - 1, self.mid - 1, -1):
            if projection_sites and site == projection_sites[-1]:
                del projection_sites[-1]
                spin_site = projection_spins.pop()
                state_site = self.state[site][spin_site, :, :]
                right_matrix = state_site @ right_matrix * np.sqrt(2)
            else:
                state_site = np.einsum("xab,bc->xac", self.state[site], right_matrix)
                D0, D1, D2 = state_site.shape
                state_site = state_site.transpose(1, 0, 2).reshape(D1, D0 * D2)
                U, S, Vh = np.linalg.svd(state_site, full_matrices=False)
                right_matrix = U @ np.diag(S)

        mid_matrix = left_matrix @ right_matrix / np.sqrt(self.weight)
        U, S, Vh = np.linalg.svd(mid_matrix, full_matrices=False)
        negativity = np.sum(S) ** 2 / 2 - 1 / 2
        return negativity

    @staticmethod
    def entropy(density_matrix):
        eigs, _ = np.linalg.eigh(density_matrix)
        eigs = eigs[eigs > 10 ** (-10)]
        entropy = - eigs @ np.log(eigs)
        return entropy


def compute(pre_file_name, i_list):
    start = time.time()
    cmi_list = []
    negativity_list = []
    for i in i_list:
        cmi = CMI_MC(pre_file_name + str(i_list[i]))
        cmi.fast_calculation_negativity_cmi()
        cmi_list.append(cmi.cmi)
        negativity_list.append(cmi.negativity)
        print(time.time() - start)
    return cmi_list, negativity_list



cmi_list, negativity_list = compute("data/zxz_model/set_8/tenpy_result_", range(5))
print(cmi_list)
print(negativity_list)
with shelve.open("data/zxz_model/set_8/cmi_data") as db:
    db["cmi_list"] = cmi_list
    db["negativity_list"] = negativity_list



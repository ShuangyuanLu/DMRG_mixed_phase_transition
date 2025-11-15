# GPU-accelerated version: all einsum / linalg run on GPU via CuPy when available.
import numpy as _np
try:
    import cupy as _cp
    xp = _cp
    ON_GPU = True
except Exception:
    xp = _np
    ON_GPU = False

import shelve
# from ZXZ_chain import ZXZ_chain   # not needed anymore; we compute norm on GPU directly
import random
import matplotlib.pyplot as plt
import time
import math

import cupy as cp, gc, time


# ---------- helpers for device conversion & scalars ----------
def to_xp(a):
    """Move numpy array/list to xp (GPU if CuPy present)."""
    if isinstance(a, list):
        return [to_xp(x) for x in a]
    if ON_GPU and isinstance(a, _np.ndarray):
        return xp.asarray(a)
    return a

def to_numpy_scalar(x):
    """Convert xp scalar/0-d array to Python float."""
    if ON_GPU:
        if isinstance(x, xp.ndarray):
            return float(x.get().item())
        try:
            return float(_cp.asnumpy(x))
        except Exception:
            return float(x)
    else:
        if isinstance(x, _np.ndarray):
            return float(x.item())
        return float(x)

def svd_xp(mat, full_matrices=False):
    return xp.linalg.svd(mat, full_matrices=full_matrices)

def eigh_xp(mat):
    return xp.linalg.eigh(mat)


class CMI_MC:
    def __init__(self, file_name, model_type="zxz"):
        # define Pauli / Id on xp
        self.id = xp.array([[1, 0], [0, 1]])
        self.sx = xp.array([[0, 1], [1, 0]])
        self.sz = xp.array([[1, 0], [0, -1]])

        # load CPU arrays from shelve, then move to GPU
        with shelve.open(file_name) as db:
            state = db["state"]           # list of numpy arrays
            parameters = db["parameters"] # dict

        # state[i]: (Dl, d, Dr) originally in your pipeline becomes (d, Dl, Dr)
        for i in range(len(state)):
            state[i] = state[i].transpose(1, 0, 2)  # (d, Dl, Dr)

        # move state to xp (GPU)
        state = to_xp(state)

        # Build X-even / X-odd 4×4 blocks on xp
        X_even_matrix = xp.zeros([4, 4, 2, 2], dtype=self.id.dtype)
        X_odd_matrix = xp.zeros([4, 4, 2, 2], dtype=self.id.dtype)
        X_even_matrix[0, 0, :, :] = xp.eye(2, dtype=self.id.dtype)
        X_even_matrix[1, 1, :, :] = self.sx
        X_even_matrix[2, 2, :, :] = xp.eye(2, dtype=self.id.dtype)
        X_even_matrix[3, 3, :, :] = self.sx

        X_odd_matrix[0, 0, :, :] = xp.eye(2, dtype=self.id.dtype)
        X_odd_matrix[1, 1, :, :] = xp.eye(2, dtype=self.id.dtype)
        X_odd_matrix[2, 2, :, :] = self.sx
        X_odd_matrix[3, 3, :, :] = self.sx

        boundary = xp.ones([1, 4], dtype=self.id.dtype)

        # apply alternating X blocks (all on GPU)
        for i in range(len(state)):
            state_i = state[i]
            X_matrix = X_even_matrix if (i % 2 == 0) else X_odd_matrix

            if i == 0:
                # X_matrix_1[a,b,x,y], boundary[c,a] -> [c,b,x,y] / 2
                X_matrix_1 = xp.einsum("abxy,ca->cbxy", X_matrix, boundary) / 2.0
            elif i == len(state) - 1:
                X_matrix_1 = xp.einsum("abxy,cb->acxy", X_matrix, boundary) / 2.0
            else:
                X_matrix_1 = X_matrix

            # state_i[x,a,b], X_matrix_1[c,d,x,y] -> [y, a, c, b, d]
            state_i = xp.einsum("xab,cdxy->yacbd", state_i, X_matrix_1)
            D0, D1, D2, D3, D4 = state_i.shape
            state_i = state_i.reshape(D0, D1 * D2, D3 * D4)  # (d, Dl, Dr)
            state[i] = state_i

        # apply zx rotation on odd sites (GPU)
        zx_rotation = xp.array([[1, 1], [1, -1]], dtype=self.id.dtype) / xp.sqrt(2.0)
        for i in range(len(state)):
            if i % 2 == 1:
                state[i] = xp.einsum("iab,ij->jab", state[i], zx_rotation)

        # GPU norm via sequential MPS contraction (no ZXZ_chain dependency)
        norm = xp.sqrt(self._mps_norm(state))
        print("norm:", to_numpy_scalar(norm))
        state[0] = state[0] / norm

        random.seed(0)

        self.state = state
        self.L = len(state)
        self.x_0 = self.L // 8 * 2
        self.x_1 = self.L // 8 * 6

        self.projection_sites = [2 * i + 1 for i in range(self.L // 2)]
        self.n_projection = len(self.projection_sites)

        self.operator_right = None
        self.operator_left = None
        self.operator_left_record = None
        self.operator_right_record = None

        # self.mc_state = [random.randint(0, 1) for _ in range(self.L // 2)]
        self.mc_state = [0 for _ in range(self.L // 2)]  # keep even charge
        self.weight = self.contraction_after_projection()
        self.operator_left = self.operator_left_record
        self.operator_right = self.operator_right_record

        self.n_mc = 1
        self.n_measure = 1
        self.cmi = []
        self.negativity = []
        self.mid = self.L // 2

    # ======== core GPU contractions ========

    def _mps_norm(self, state_list):
        """Exact ⟨ψ|ψ⟩ via left->right contraction on GPU: state[i] has shape (d, Dl, Dr)."""
        env = xp.ones((1, 1), dtype=state_list[0].dtype)
        for i in range(len(state_list)):
            A = state_list[i]              # (d, Dl, Dr)
            # env[cd] = sum_xb env[cb] * conj(A[x,b,d]) * A[x,b,c] (two-step einsums)
            env = xp.einsum("cd,xdb->xcb", env, A)                 # (x,c,b)
            env = xp.einsum("xcb,xca->ab", env, A.conj())          # (a,b)
        # scalar at (0,0)
        return env[0, 0]

    def fast_calculation_negativity_cmi(self):
        self.cmi = self.contraction_with_two_sites_left()
        self.negativity = self.middle_canonical_form()
        print("cmi:", to_numpy_scalar(self.cmi))
        print("negativity:", to_numpy_scalar(self.negativity))

    def contraction_after_projection(self):
        operator_right = xp.ones([1, 1], dtype=self.state[0].dtype)
        operator_left = xp.ones([1, 1], dtype=self.state[0].dtype)
        projection_sites = self.projection_sites.copy()
        projection_spins = self.mc_state.copy()

        for site in range(self.x_0):
            if projection_sites and site == projection_sites[0]:
                del projection_sites[0]
                spin_site = projection_spins.pop(0)
                state_site = self.state[site][spin_site, :, :]  # (Dl, Dr)
                operator_left = xp.einsum("cd,db->cb", operator_left, state_site)
                operator_left = xp.einsum("cb,ca->ab", operator_left, state_site.conj()) * 2.0
            else:
                operator_left = xp.einsum("cd,xdb->xcb", operator_left, self.state[site])
                operator_left = xp.einsum("xcb,xca->ab", operator_left, self.state[site].conj())
        self.operator_left_record = operator_left

        for site in range(self.L - 1, self.x_0 - 1, -1):
            if projection_sites and site == projection_sites[-1]:
                del projection_sites[-1]
                spin_site = projection_spins.pop()
                state_site = self.state[site][spin_site, :, :]  # (Dl, Dr)
                operator_right = xp.einsum("ab,db->ad", operator_right, state_site)
                operator_right = xp.einsum("ad,ca->cd", operator_right, state_site.conj()) * 2.0
            else:
                operator_right = xp.einsum("ab,xdb->xad", operator_right, self.state[site])
                operator_right = xp.einsum("xad,xca->cd", operator_right, self.state[site].conj())

            if site == self.x_1 + 1:
                self.operator_right_record = operator_right

        result = xp.einsum("ab,ab->", operator_left, operator_right)  # scalar
        return to_numpy_scalar(result) if not isinstance(result, (float, int)) else result

    def contraction_with_two_sites_left(self):
        """
        Compute CMI for the two central sites using precomputed left/right envs on GPU.
        """
        projection_sites = [self.projection_sites[i] for i in range(len(self.projection_sites))
                            if self.x_0 < self.projection_sites[i] < self.x_1]
        projection_spins = [self.mc_state[i] for i in range(len(self.mc_state))
                            if self.x_0 < self.projection_sites[i] < self.x_1]

        operator_left = xp.einsum("cd,xdb->xcb", self.operator_left, self.state[self.x_0])
        operator_left = xp.einsum("xcb,yca->xyab", operator_left, self.state[self.x_0].conj())

        operator_right = xp.einsum("ab,wdb->wad", self.operator_right, self.state[self.x_1])
        operator_right = xp.einsum("wad,zca->wzcd", operator_right, self.state[self.x_1].conj())

        for site in range(self.x_1 - 1, self.x_0, -1):
            if projection_sites and site == projection_sites[-1]:
                del projection_sites[-1]
                spin_site = projection_spins.pop()
                state_site = self.state[site][spin_site, :, :]
                operator_right = xp.einsum("wzab,db->wzad", operator_right, state_site)
                operator_right = xp.einsum("wzad,ca->wzcd", operator_right, state_site.conj()) * 2.0
            else:
                operator_right = xp.einsum("wzab,xdb->wzxad", operator_right, self.state[site])
                operator_right = xp.einsum("wzxad,xca->wzcd", operator_right, self.state[site].conj())

        two_site_density_matrix = xp.einsum("xyab,wzab->xywz", operator_left, operator_right)
        two_site_density_matrix = two_site_density_matrix / self.weight

        rho_2 = two_site_density_matrix.transpose(0, 2, 1, 3).reshape(4, 4)
        rho_1_A = xp.einsum("xyww->xy", two_site_density_matrix)
        rho_1_B = xp.einsum("xxwz->wz", two_site_density_matrix)
        cmi = CMI_MC.entropy(rho_1_A) + CMI_MC.entropy(rho_1_B) - CMI_MC.entropy(rho_2)
        return cmi

    def middle_canonical_form(self):
        projection_sites = self.projection_sites.copy()
        projection_spins = self.mc_state.copy()

        left_matrix = xp.ones([1, 1], dtype=self.state[0].dtype)
        for site in range(self.mid):
            if projection_sites and site == projection_sites[0]:
                del projection_sites[0]
                spin_site = projection_spins.pop(0)
                state_site = self.state[site][spin_site, :, :]
                left_matrix = left_matrix @ state_site * xp.sqrt(2.0)
            else:
                state_site = xp.einsum("xab,ca->xcb", self.state[site], left_matrix)
                D0, D1, D2 = state_site.shape
                state_site = state_site.reshape(D0 * D1, D2)
                U, S, Vh = svd_xp(state_site, full_matrices=False)
                left_matrix = xp.diag(S) @ Vh

        right_matrix = xp.ones([1, 1], dtype=self.state[0].dtype)
        for site in range(self.L - 1, self.mid - 1, -1):
            if projection_sites and site == projection_sites[-1]:
                del projection_sites[-1]
                spin_site = projection_spins.pop()
                state_site = self.state[site][spin_site, :, :]
                right_matrix = state_site @ right_matrix * xp.sqrt(2.0)
            else:
                state_site = xp.einsum("xab,bc->xac", self.state[site], right_matrix)
                D0, D1, D2 = state_site.shape
                state_site = state_site.transpose(1, 0, 2).reshape(D1, D0 * D2)
                U, S, Vh = svd_xp(state_site, full_matrices=False)
                right_matrix = U @ xp.diag(S)

        mid_matrix = left_matrix @ right_matrix / xp.sqrt(self.weight)
        U, S, Vh = svd_xp(mid_matrix, full_matrices=False)
        negativity = (xp.sum(S) ** 2) / 2.0 - 0.5
        return negativity

    @staticmethod
    def entropy(density_matrix):
        eigs, _ = eigh_xp(density_matrix)
        eigs = eigs[eigs > 10 ** (-10)]
        return float((_np.array(eigs.get()) if ON_GPU else eigs).dot(_np.log(_np.array(eigs.get()) if ON_GPU else eigs))) * (-1.0)


def compute(pre_file_name, i_list):
    mp  = cp.get_default_memory_pool()
    pmp = cp.get_default_pinned_memory_pool()

    start = time.time()
    cmi_list = []
    negativity_list = []
    for i in i_list:
        cmi = CMI_MC(pre_file_name + str(i_list[i]))
        cmi.fast_calculation_negativity_cmi()
        cmi_list.append(to_numpy_scalar(cmi.cmi))
        negativity_list.append(to_numpy_scalar(cmi.negativity))

        # cleanup to avoid pool bloat between runs
        del cmi
        cp._default_memory_pool.free_all_blocks()
        cp._default_pinned_memory_pool.free_all_blocks()
        gc.collect()

        cp.cuda.runtime.deviceSynchronize()
        print(time.time() - start)
    return cmi_list, negativity_list


if __name__ == "__main__":
    cmi_list, negativity_list = compute("data/zxz_model/set_2/tenpy_result_", range(8))
    print(cmi_list)
    print(negativity_list)
    with shelve.open("data/zxz_model/set_2/cmi_data") as db:
        db["cmi_list"] = cmi_list
        db["negativity_list"] = negativity_list

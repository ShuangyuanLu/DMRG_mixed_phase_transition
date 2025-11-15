from blas_single import set_single_thread_env
set_single_thread_env()
# zxz_dmrg_tenpy_clean.py
import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain, Lattice
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import time
import shelve


class ZXZ_gapless_Chain(CouplingMPOModel):
    default_lattice = Chain  # helps the base class wire defaults

    def init_sites(self, model_params):
        # spin-1/2 without charge conservation (X terms break Sz)
        site = SpinHalfSite(conserve=None)
        # add true Pauli operators so we don't track factors of 2
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1], [1, 0]]) # one 1j different from real Y Pauli matrix, made up by one more minus sign on YXY term
        Z = np.array([[1, 0], [0, -1]])
        site.add_op("X", X); site.add_op("Y", Y); site.add_op("Z", Z)
        return site

    def init_lattice(self, p):
        L = int(p.get("L", 20))        # total sites
        if L % 2 != 0:
            raise ValueError("L must be even for a 2-site unit cell.")
        bc, bc_MPS = p.get("bc", "open"), p.get("bc_MPS", "finite")

        site = self.init_sites(p)
        L_cells = L // 2               # number of [A,B] cells

        # 1D Bravais vector and A/B positions (purely cosmetic)
        basis     = np.array([[1.0]])          # one spatial dimension
        positions = np.array([[0.0], [0.5]])   # A at 0, B at +0.5

        # Build a 2-site unit cell directly with Lattice (no SimpleLattice wrapping)
        return Lattice([L_cells], [site, site], bc=bc, bc_MPS=bc_MPS,
                       basis=basis, positions=positions)

    def init_terms(self, p):
        J = float(p.get("J", 1.0))
        h = float(p.get("h", 0.0))
        K = float(p.get("K", 1.0))

        # ---------- ZXZ everywhere ----------
        # even-start (A_i, B_i, A_{i+1}) -> (u,dx): (0,0),(1,0),(0,1)
        self.add_multi_coupling(-J, [("Z", 0, 0), ("X", 0, 1), ("Z", 1, 0)],
                                category="ZXZ_even_start")
        # odd-start  (B_i, A_{i+1}, B_{i+1}) -> (1,0),(0,1),(1,1)
        self.add_multi_coupling(-J, [("Z", 0, 1), ("X", 1, 0), ("Z", 1, 1)],
                                category="ZXZ_odd_start")

        # ---------- X field on every site ----------
        self.add_onsite(-h, 0, "X", category="X_field")  # A
        self.add_onsite(-h, 1, "X", category="X_field")  # B

        # ---------- YXY only on even starts (A-starts) ----------
        if K != 0.0:
            self.add_multi_coupling(K, [("Y", 0, 0), ("X", 0, 1), ("Y", 1, 0)],
                                    category="YXY_even_start")
            # coefficient +K is -(-K), first minus sign is use do make up our definition of Y, -K is the physical model coefficient


def run_dmrg_zxz_gapless(L=50, J=1.0, h=0.5, K=1.0, chi_max=128, sweeps=20, seed=11):
    parameters = {"L": L, "J": J, "h": h, "K": K, "bc": "open", "bc_MPS": "finite"}
    model = ZXZ_gapless_Chain(parameters)

    # simple product start |↑...↑>
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up"] * L, bc=model.lat.bc_MPS)

    dmrg_opts = {
        "mixer": True,
        "max_sweeps": sweeps,
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
        # Optional: smoother growth
        # "chi_list": [(0, min(64, chi_max)), (4, min(128, chi_max)), (8, chi_max)],
    }

    info = dmrg.run(psi, model, dmrg_opts)
    E = info["E"] if isinstance(info, dict) else info[0]
    e0 = E / L

    # <Z_i X_{i+1} Z_{i+2}>
    k_vals = []
    for i in range(0, L - 2):
        val = psi.expectation_value(["Z", "X", "Z"], sites=[i, i + 1, i + 2])
        k_vals.append(np.real_if_close(val))
    k_vals = np.asarray(k_vals)
    k_bulk = float(k_vals.mean()) if k_vals.size else float("nan")

    mx = float(np.mean(psi.expectation_value("X")))
    mz = float(np.mean(psi.expectation_value("Z")))

    Bs_np = [psi.get_B(i).to_ndarray() for i in range(psi.L)]  # list of npc.Array

    print(f"--- ZXZ DMRG (L={L}, J={J}, h={h}) ---")
    print(f"Ground-state energy: E = {E:.12f}  (e0 = {e0:.12f} per site)")
    print(f"<Z X Z> (bulk avg): {k_bulk:.8f}")
    print(f"<X> avg: {mx:.8f}   <Z> avg: {mz:.8f}")
    print(f"Final bond dims     : {psi.chi}")

    return {
        "E": float(E),
        "e0": float(e0),
        "K_bulk": k_bulk,
        "mx": mx,
        "mz": mz,
        "chi": list(psi.chi),
        "k_profile": k_vals,
        "state": Bs_np,
        "parameters": parameters
    }


# if __name__ == "__main__":
#     start_time = time.perf_counter()
#     result = run_dmrg_zxz_gapless(L=20, J=1.0, h=0.2, K=0.5, chi_max=20, sweeps=12)
#     end_time = time.perf_counter()
#
#     for x in result["state"]:
#         print(x.shape)
#     # print(result["state"])
#     print("time:", round(end_time - start_time, 2))

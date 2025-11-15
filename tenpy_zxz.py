from blas_single import set_single_thread_env
set_single_thread_env()
# zxz_dmrg_tenpy_clean.py
import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import time
import shelve


class ZXZChain(CouplingMPOModel):
    r"""
    1D ZXZ (cluster-field) model:
        H = -J * sum_i Z_i X_{i+1} Z_{i+2}  -  h * sum_i X_i
    Open boundary conditions; spin-1/2 sites (no symmetry conservation).
    """

    default_lattice = Chain  # helps the base class wire defaults

    def init_sites(self, model_params):
        # spin-1/2 without charge conservation (X terms break Sz)
        site = SpinHalfSite(conserve=None)
        # add true Pauli operators so we don't track factors of 2
        site.add_op("X", 2.0 * site.Sx)
        site.add_op("Z", 2.0 * site.Sz)
        return [site]

    def init_lattice(self, model_params):
        L = int(model_params.get("L", 50))
        bc = model_params.get("bc", "open")
        bc_MPS = model_params.get("bc_MPS", "finite")
        site = self.init_sites(model_params)[0]
        return Chain(L, site, bc=bc, bc_MPS=bc_MPS)

    def init_terms(self, model_params):
        J = float(model_params.get("J", 1.0))
        h = float(model_params.get("h", 0.5))

        # three-body ZXZ: sum_i Z_i X_{i+1} Z_{i+2}
        # add_multi_coupling(strength, [(op, dx, u), ...], ...)
        self.add_multi_coupling(
            -J,
            [("Z", 0, 0), ("X", +1, 0), ("Z", +2, 0)],
            category="ZXZ",
        )

        # onsite X field
        # add_onsite(strength, u, opname, ...)
        self.add_onsite(-h, 0, "X", category="X_field")


def run_dmrg_zxz(L=50, J=1.0, h=0.5, chi_max=128, sweeps=20, seed=0):
    parameters = {"L": L, "J": J, "h": h, "bc": "open", "bc_MPS": "finite"}
    model = ZXZChain(parameters)

    # simple product start |↑...↑>
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up"] * L, bc=model.lat.bc_MPS)

    dmrg_opts = {
        "mixer": True,
        "max_sweeps": sweeps,
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
        # "verbose": True,
        # "random_seed": seed,
        # Optional: smoother growth
        # "chi_list": [(0, min(64, chi_max)), (4, min(128, chi_max)), (8, chi_max)],
    }

    info = dmrg.run(psi, model, dmrg_opts)
    E = info["E"] if isinstance(info, dict) else info[0]
    e0 = E / L

    mx = float(np.mean(psi.expectation_value("X")))
    mz = float(np.mean(psi.expectation_value("Z")))

    Bs_np = [psi.get_B(i).to_ndarray() for i in range(psi.L)]  # list of npc.Array

    print(f"--- ZXZ DMRG (L={L}, J={J}, h={h}) ---")
    print(f"Ground-state energy: E = {E:.12f}  (e0 = {e0:.12f} per site)")
    print(f"<X> avg: {mx:.8f}   <Z> avg: {mz:.8f}")
    print(f"Final bond dims     : {psi.chi}")

    return {
        "E": float(E),
        "e0": float(e0),
        "mx": mx,
        "mz": mz,
        "chi": list(psi.chi),
        "state": Bs_np,
        "parameters": parameters
    }


# if __name__ == "__main__":
#     start_time = time.perf_counter()
#     result = run_dmrg_zxz(L=20, J=1.0, h=0.2, chi_max=20, sweeps=12)
#     end_time = time.perf_counter()
#     # print(result["state"])
#     print("time:", round(end_time - start_time, 2))

from blas_single import set_single_thread_env
set_single_thread_env()

import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import time
import shelve


class Spin_1_Chain(CouplingMPOModel):
    r"""
    1D spin-1 Heisenberg model with single-ion anisotropy:
        H = J * sum_{i} ( S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z )
          + h * sum_{i} (S_i^z)^2

    Open boundary conditions; U(1) Sz conservation is enabled (no transverse fields).
    Convention: +J is antiferromagnetic.
    """

    default_lattice = Chain  # helps the base class wire defaults

    def init_sites(self, model_params):
        # No symmetry to avoid missing-operator issues
        site = SpinSite(S=1, conserve=None)

        # Core spin ops exist already: "Sz", "Sp", "Sm"
        Sz = site.get_op("Sz")

        # Build (Sz)^2 as an onsite operator using npc.tensordot
        # Contract ket leg 'p' of the left with bra leg 'p*' of the right
        Sz2 = npc.tensordot(Sz, Sz, axes=(("p",), ("p*",)))  # labels remain ('p*','p')
        site.add_op("Sz2", Sz2)

        return [site]

    def init_lattice(self, model_params):
        L = int(model_params.get("L", 50))
        bc = model_params.get("bc", "open")
        bc_MPS = model_params.get("bc_MPS", "finite")
        site = self.init_sites(model_params)[0]
        return Chain(L, site, bc=bc, bc_MPS=bc_MPS)

    def init_terms(self, model_params):
        # Mark params as used (avoids "unused options" warnings)
        J = float(model_params.get("J", 1.0))
        h = float(model_params.get("h", 0.5))

        # Heisenberg: SzSz + 1/2(SpSm + SmSp) on nearest neighbors
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(J / 2.0, u1, "Sp", u2, "Sm", dx, plus_hc=True)
            self.add_coupling(J, u1, "Sz", u2, "Sz", dx)

        # Single-ion anisotropy h (Sz)^2
        self.add_onsite(h, 0, "Sz2", category="Sz2")



def run_dmrg_spin1_heisenberg(L=50, J=1.0, h=0., chi_max=64, sweeps=12):
    parameters = {"L": L, "J": J, "h": h, "bc": "open", "bc_MPS": "finite"}
    model = Spin_1_Chain(parameters)

    # simple product state in m=0 (integer label)
    site0 = model.lat.mps_sites()[0]
    labels_map = getattr(site0, "state_labels", {})
    start_label = 0 if isinstance(labels_map, dict) and 0 in labels_map else list(labels_map)[0]
    psi = MPS.from_product_state(model.lat.mps_sites(), [start_label]*L, bc=model.lat.bc_MPS)

    dmrg_opts = {
        "mixer": True,
        "max_sweeps": sweeps,
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
    }
    info = dmrg.run(psi, model, dmrg_opts)
    psi.canonical_form()  # tidy up canonicalization
    Bs_np = [psi.get_B(i).to_ndarray() for i in range(psi.L)]  # list of npc.Array

    E = info["E"] if isinstance(info, dict) else info[0]
    e0 = E / L
    print(f"E = {E:.12f}  (e0 = {e0:.12f})")
    print(f"Final bond dims     : {psi.chi}")
    return {"E": float(E), "e0": float(e0), "state": Bs_np, "parameters": parameters}


# if __name__ == "__main__":
#     start_time = time.perf_counter()
#     result = run_dmrg_spin1_heisenberg(L=50, J=1.0, h=0.95, chi_max=50, sweeps=5)
#     end_time = time.perf_counter()
#     print("E =", result["E"], "e0 =", result["e0"])
#     print("time:", round(end_time - start_time, 2))



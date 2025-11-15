# zxz3_dmrg_tenpy_clean.py
import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinSite  # 3-level local Hilbert space (S=1)
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import time


def clock_ops_Z3():
    """Return generalized Pauli matrices X3, Z3 for Z3 clock model."""
    omega = np.exp(2j * np.pi / 3.0)
    Z3 = np.diag([1.0, omega, omega**2])  # diag(1, ω, ω^2)
    X3 = np.array(
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0]], dtype=complex
    )  # cyclic shift: |j> -> |j+1 mod 3|
    return X3, Z3


class ZXZ3Chain(CouplingMPOModel):
    r"""
    1D Z3-clock 'ZXZ' (cluster-field) model (open boundary conditions):
        H = -J * sum_i [ Z_i X_{i+1} Z_{i+2}^\dagger + h.c. ]
            -h * sum_i [ X_i + X_i^\dagger ]

    Local space: qutrit (dim=3). No explicit symmetry conservation here.
    """

    default_lattice = Chain

    def init_sites(self, model_params):
        # 3-level site via SpinSite(S=1); we won't use the SU(2) operators.
        # We simply attach our custom Z3/X3.
        site = SpinSite(S=1, conserve=None)  # basis m=-1,0,+1 (dim=3)
        X3, Z3 = clock_ops_Z3()
        site.add_op("X3", X3)
        site.add_op("X3dag", X3.conj().T)
        site.add_op("Z3", Z3)
        site.add_op("Z3dag", Z3.conj().T)
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

        # Three-body U_i = Z_i X_{i+1} Z_{i+2}^\dagger  (and add H.c. for hermiticity)
        # add_multi_coupling(strength, [(opname, dx, u_idx), ...], category=...)
        # Term: -J * sum (U + U^\dagger)
        self.add_multi_coupling(
            -J,
            [("Z3", 0, 0), ("X3", +1, 0), ("Z3dag", +2, 0)],
            category="ZXZ3"
        )
        self.add_multi_coupling(
            -J,
            [("Z3dag", 0, 0), ("X3dag", +1, 0), ("Z3", +2, 0)],
            category="ZXZ3"
        )

        # Onsite 'field' in X3-direction: -h * (X3 + X3^\dagger)
        self.add_onsite(-h, 0, "X3", category="X3_field")
        self.add_onsite(-h, 0, "X3dag", category="X3_field")


def run_dmrg_zxz3(L=50, J=1.0, h=0.5, chi_max=128, sweeps=20, seed=11, start_state="0"):
    """
    Run DMRG for the Z3 cluster-field model.

    start_state: one of {'up','0','down'} for SpinSite(S=1) product states.
                 (These map to m=+1,0,-1; any is fine since we use custom ops.)
    """
    parameters = {"L": L, "J": J, "h": h, "bc": "open", "bc_MPS": "finite"}
    model = ZXZ3Chain(parameters)

    # Simple product start |m=start_state>^L  (choices: 'up', '0', 'down')
    psi = MPS.from_product_state(model.lat.mps_sites(), [start_state] * L, bc=model.lat.bc_MPS)

    dmrg_opts = {
        "mixer": True,
        "max_sweeps": sweeps,
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
        "verbose": True,
        "random_seed": seed,
        # Optional smoother growth:
        # "chi_list": [(0, min(64, chi_max)), (4, min(128, chi_max)), (8, chi_max)],
    }

    info = dmrg.run(psi, model, dmrg_opts)
    E = info["E"] if isinstance(info, dict) else info[0]
    e0 = E / L

    # Expectation of the (Hermitian) cluster term density:
    # k_i = Re( <Z_i X_{i+1} Z_{i+2}^\dagger> )  (since we added H.c. separately in H)
    k_vals = []
    for i in range(0, L - 2):
        val = psi.expectation_value(["Z3", "X3", "Z3dag"], sites=[i, i + 1, i + 2])
        k_vals.append(np.real_if_close(val))
    k_vals = np.asarray(k_vals, dtype=float)
    k_bulk = float(k_vals.mean()) if k_vals.size else float("nan")

    # Onsite order parameters (complex in general). Report real part and magnitude.
    x_vals = psi.expectation_value("X3")
    x_re = float(np.real(np.mean(x_vals)))
    x_abs = float(np.mean(np.abs(x_vals)))

    z_vals = psi.expectation_value("Z3")
    z_re = float(np.real(np.mean(z_vals)))
    z_abs = float(np.mean(np.abs(z_vals)))

    Bs_np = [psi.get_B(i).to_ndarray() for i in range(psi.L)]

    print(f"--- Z3 ZXZ DMRG (L={L}, J={J}, h={h}) ---")
    print(f"Ground-state energy: E = {E:.12f}  (e0 = {e0:.12f} per site)")
    print(f"Re<Z X Z^†> (bulk avg): {k_bulk:.8f}")
    print(f"<X3> avg: Re={x_re:.8f}  |.|={x_abs:.8f}")
    print(f"<Z3> avg: Re={z_re:.8f}  |.|={z_abs:.8f}")
    print(f"Final bond dims       : {psi.chi}")

    return {
        "E": float(E),
        "e0": float(e0),
        "K_bulk": k_bulk,
        "X3_Re": x_re,
        "X3_abs": x_abs,
        "Z3_Re": z_re,
        "Z3_abs": z_abs,
        "chi": list(psi.chi),
        "k_profile": k_vals,
        "state": Bs_np,
        "parameters": parameters,
    }


'''
if __name__ == "__main__":
    start_time = time.perf_counter()
    result = run_dmrg_zxz3(L=24, J=1.0, h=0.5, chi_max=128, sweeps=12, seed=11, start_state="0")
    end_time = time.perf_counter()
    print("e0:", result["e0"])
    print("time:", round(end_time - start_time, 2), "s")
'''

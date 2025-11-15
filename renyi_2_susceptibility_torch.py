import torch
import numpy as np


def renyi_2_susceptibility_torch(state, site_list, operator, quantum_channel, quantum_channel_list):
    """
    Torch version (GPU-ready) of renyi_2_susceptibility.
    - state: list of site tensors; each can be torch.Tensor or np.ndarray with shapes
             matching your original einsums (e.g. (x,g,c) and its conjugate used accordingly).
    - operator: (s, s) array/tensor
    - quantum_channel: (w,z,x,y) array/tensor
    - site_list: iterable of sites to apply 'operator' (same semantics as before)
    - quantum_channel_list: iterable of sites where the quantum channel acts
    Returns: Python float (susceptibility)
    """
    # --- unify device & dtype ---
    def to_torch(x, device, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        else:
            return torch.as_tensor(x, device=device, dtype=dtype)

    # Infer device/dtype from first state tensor (or pick defaults)
    first = state[0] if len(state) > 0 else None
    if isinstance(first, torch.Tensor):
        device = first.device
        # preserve complex dtype if present; else use complex128 by default
        dtype = first.dtype if first.is_complex() else torch.complex128
    else:
        # default to GPU if available; otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.complex128

    # Convert state list
    state_t = [to_torch(s, device, dtype) for s in state]

    # Convert operator & quantum channel
    operator_t = to_torch(operator, device, dtype)
    qc_t = to_torch(quantum_channel, device, dtype)

    # Precompute sets for fast membership
    site_set = set(site_list)
    qc_set = set(quantum_channel_list)

    L = len(state_t)

    # --- small helper constructors on correct device/dtype ---
    eye2 = torch.eye(2, device=device, dtype=dtype)
    zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
    ones  = lambda *shape: torch.ones(*shape, device=device, dtype=dtype)

    # Tensors initialized exactly as in NumPy version
    operator_right = zeros(1, 1, 2, 1, 1)
    operator_right[0, 0, 1, 0, 0] = 1

    operator_right_2 = zeros(1, 1, 2, 2, 1, 1)
    operator_right_2[0, 0, 1, 1, 0, 0] = 1

    s = operator_t.shape[0]
    m_tensor = zeros(s, s, s, s, 2, 2)
    # m[:, :, :, :, 0,0] and m[:, :, :, :, 1,1] are I⊗I with indices permuted as "wzxy"
    base = torch.einsum("xw,yz->wzxy", eye2, eye2)
    m_tensor[:, :, :, :, 0, 0] = base
    m_tensor[:, :, :, :, 1, 1] = base
    # m[:, :, :, :, 0,1] is operator ⊗ operator.conj() with same permutation
    m_tensor[:, :, :, :, 0, 1] = torch.einsum("xw,yz->wzxy", operator_t, operator_t.conj())

    operator_right_0 = ones(1, 1, 1, 1)
    norm_list = torch.zeros(L, device=device, dtype=operator_right_0.dtype)

    # --- right environment for normalization ---
    for site in range(L - 1, -1, -1):
        # "abcd,xgc->abxgd"
        operator_right_0 = torch.einsum("abcd,xgc->abxgd", operator_right_0, state_t[site])
        # "abxgd,yhd->abxghy"
        operator_right_0 = torch.einsum("abxgd,yhd->abxghy", operator_right_0, state_t[site].conj())
        if site in qc_set:
            # "abwghz,wzxy->abxghy"
            operator_right_0 = torch.einsum("abwghz,wzxy->abxghy", operator_right_0, qc_t)
            # "abwghz,xywz->abxghy"
            operator_right_0 = torch.einsum("abwghz,xywz->abxghy", operator_right_0, qc_t.conj())
        # "abxghy,yea->ebxgh"
        operator_right_0 = torch.einsum("abxghy,yea->ebxgh", operator_right_0, state_t[site])
        # "ebxgh,xfb->efgh"
        operator_right_0 = torch.einsum("ebxgh,xfb->efgh", operator_right_0, state_t[site].conj())

        norm_val = torch.linalg.norm(operator_right_0)
        # guard against zero norm to avoid NaNs (rare, but safer)
        norm_val = norm_val + (norm_val == 0).to(norm_val.dtype) * 1e-30
        norm_list[site] = norm_val
        operator_right_0 = operator_right_0 / norm_val

    # --- operator_right contraction ---
    for site in range(L - 1, -1, -1):
        operator_right = torch.einsum("abicd,xgc->abixgd", operator_right, state_t[site])
        operator_right = torch.einsum("abixgd,yhd->abixghy", operator_right, state_t[site].conj())
        if site in qc_set:
            operator_right = torch.einsum("abiwghz,wzxy->abixghy", operator_right, qc_t)
        if site in site_set:
            operator_right = torch.einsum("abiwghz,wzxyji->abjxghy", operator_right, m_tensor)
        if site in qc_set:
            operator_right = torch.einsum("abjwghz,xywz->abjxghy", operator_right, qc_t.conj())
        operator_right = torch.einsum("abjxghy,yea->ebjxgh", operator_right, state_t[site])
        operator_right = torch.einsum("ebjxgh,xfb->efjgh", operator_right, state_t[site].conj())

        operator_right = operator_right / norm_list[site]

    # --- operator_right_2 contraction ---
    for site in range(L - 1, -1, -1):
        operator_right_2 = torch.einsum("abikcd,xgc->abikxgd", operator_right_2, state_t[site])
        operator_right_2 = torch.einsum("abikxgd,yhd->abikxghy", operator_right_2, state_t[site].conj())
        if site in qc_set:
            operator_right_2 = torch.einsum("abikwghz,wzxy->abikxghy", operator_right_2, qc_t)
        if site in site_set:
            operator_right_2 = torch.einsum("abikwghz,wzxyji->abjkxghy", operator_right_2, m_tensor)
            operator_right_2 = torch.einsum("abjkwghz,wzxylk->abjlxghy", operator_right_2, m_tensor)
        if site in qc_set:
            operator_right_2 = torch.einsum("abjlwghz,xywz->abjlxghy", operator_right_2, qc_t.conj())
        operator_right_2 = torch.einsum("abjlxghy,yea->ebjlxgh", operator_right_2, state_t[site])
        operator_right_2 = torch.einsum("ebjlxgh,xfb->efjlgh", operator_right_2, state_t[site].conj())

        operator_right_2 = operator_right_2 / norm_list[site]

    corr   = operator_right[0, 0, 0, 0, 0] / operator_right_0[0, 0, 0, 0]
    corr_2 = operator_right_2[0, 0, 0, 0, 0, 0] / operator_right_0[0, 0, 0, 0]
    sus = (corr_2 - corr ** 2) / max(1, len(site_list))

    # return python float (real part, in case small imag noise)
    return sus.real.item()

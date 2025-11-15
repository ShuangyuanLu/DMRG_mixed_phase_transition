import torch


def renyi_2_correlator_torch(
    state,
    site_1,
    site_2,
    operator,
    quantum_channel,
    quantum_channel_list,
    device=None,
    dtype=torch.float32,
):
    # --- select device ---
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- move inputs to GPU / correct dtype ---
    torch_state = []
    for s in state:
        if isinstance(s, torch.Tensor):
            torch_state.append(s.to(device=device, dtype=dtype))
        else:
            torch_state.append(torch.as_tensor(s, device=device, dtype=dtype))
    state = torch_state

    operator = torch.as_tensor(operator, device=device, dtype=dtype)
    quantum_channel = torch.as_tensor(quantum_channel, device=device, dtype=dtype)

    # initialize environment tensors
    operator_right  = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    operator_right_0 = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)

    L = len(state)

    # loop from right to left
    for site in range(L - 1, -1, -1):
        psi = state[site]

        # --- operator_right branch ---
        operator_right = torch.einsum("abcd,xgc->abxgd", operator_right, psi)
        operator_right = torch.einsum("abxgd,yhd->abxghy", operator_right, psi)

        if site in quantum_channel_list:
            operator_right = torch.einsum("abwghz,wzxy->abxghy", operator_right, quantum_channel)

        if site == site_1 or site == site_2:
            operator_right = torch.einsum("abwghz,xw,yz->abxghy",
                                          operator_right, operator, operator)

        if site in quantum_channel_list:
            operator_right = torch.einsum("abwghz,xywz->abxghy",
                                          operator_right, quantum_channel)

        operator_right = torch.einsum("abxghy,yea->ebxgh", operator_right, psi)
        operator_right = torch.einsum("ebxgh,xfb->efgh", operator_right, psi)

        # --- operator_right_0 branch (no operator insertion) ---
        operator_right_0 = torch.einsum("abcd,xgc->abxgd", operator_right_0, psi)
        operator_right_0 = torch.einsum("abxgd,yhd->abxghy", operator_right_0, psi)

        if site in quantum_channel_list:
            operator_right_0 = torch.einsum("abwghz,wzxy->abxghy",
                                            operator_right_0, quantum_channel)
            operator_right_0 = torch.einsum("abwghz,xywz->abxghy",
                                            operator_right_0, quantum_channel)

        operator_right_0 = torch.einsum("abxghy,yea->ebxgh", operator_right_0, psi)
        operator_right_0 = torch.einsum("ebxgh,xfb->efgh", operator_right_0, psi)

        # normalization
        norm = torch.linalg.norm(operator_right_0)
        operator_right  = operator_right / norm
        operator_right_0 = operator_right_0 / norm

    # final correlator (scalar)
    corr = operator_right[0, 0, 0, 0] / operator_right_0[0, 0, 0, 0]
    return corr.item()

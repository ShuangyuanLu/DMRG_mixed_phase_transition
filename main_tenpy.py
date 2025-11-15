import os, time, multiprocessing as mp
from blas_single import set_single_thread_env
set_single_thread_env()
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass


from ZXZ_chain import ZXZ_chain
import time
import shelve
from concurrent.futures import ProcessPoolExecutor, as_completed
from tenpy_zxz import run_dmrg_zxz
from tenpy_spin_1 import run_dmrg_spin1_heisenberg
from tenpy_zxz_gapless import run_dmrg_zxz_gapless
import numpy as np
import logging
from MPS_DM_basic import MPS_basic
import re
from Spin_1_Heisenberg_chain import Spin_1_Heisenberg_chain
from zxz_gapless_chain import ZXZ_gapless_chain
from renyi_2_correlator_torch import renyi_2_correlator_torch
from renyi_2_susceptibility_torch import renyi_2_susceptibility_torch

import argparse


def strange_correlator(state, site_1, site_2, operator):
    L = len(state)
    operator_right = np.ones(1)
    operator_right_0 = np.ones(1)
    s = state[0].shape[0]
    trivial_state = np.ones(s) / np.sqrt(s)
    for site in range(L-1, -1, -1):
        if site == site_1 or site == site_2:
            state_site = np.einsum("abc,da->dbc", state[site], operator)
        else:
            state_site = state[site]
        operator_right = np.einsum("abc,c,a->b", state_site, operator_right, trivial_state)
        operator_right_0 = np.einsum("abc,c,a->b", state[site], operator_right_0, trivial_state)
        norm = np.linalg.norm(operator_right_0)
        operator_right, operator_right_0 = operator_right / norm, operator_right_0 / norm

    return operator_right[0] / operator_right_0[0]


def type_2_strange_correlator(state, site_1, site_2, operator, quantum_channel, quantum_channel_list):
    L = len(state)
    operator_right = np.ones([1, 1])
    operator_right_0 = np.ones([1, 1])
    s = state[0].shape[0]
    trivial_state = np.ones([s, s]) / s

    for site in range(L-1, -1, -1):
        operator_right = np.einsum("xab,bd->xad", state[site], operator_right)
        operator_right = np.einsum("ycd,xad->xyac", state[site].conj(), operator_right)
        operator_right_0 = np.einsum("xab,bd->xad", state[site], operator_right_0)
        operator_right_0 = np.einsum("ycd,xad->xyac", state[site].conj(), operator_right_0)
        if site in quantum_channel_list:
            operator_right = np.einsum("wzac,wzxy->xyac", operator_right, quantum_channel)
            operator_right_0 = np.einsum("wzac,wzxy->xyac", operator_right_0, quantum_channel)
        if site == site_1 or site == site_2:
            operator_right = np.einsum("wzac,xw,yz->xyac", operator_right, operator, operator.conj())
        operator_right = np.einsum("xyac,xy->ac", operator_right, trivial_state)
        operator_right_0 = np.einsum("xyac,xy->ac", operator_right_0, trivial_state)

        norm = np.linalg.norm(operator_right_0)
        operator_right, operator_right_0 = operator_right / norm, operator_right_0 / norm

    return operator_right[0, 0] / operator_right_0[0, 0]


def renyi_2_correlator(state, site_1, site_2, operator, quantum_channel, quantum_channel_list):
    L = len(state)
    operator_right = np.ones([1, 1, 1, 1])
    operator_right_0 = np.ones([1, 1, 1, 1])
    for site in range(L-1, -1, -1):
        operator_right = np.einsum("abcd,xgc->abxgd", operator_right, state[site])
        operator_right = np.einsum("abxgd,yhd->abxghy", operator_right, state[site].conjugate())
        if site in quantum_channel_list:
            operator_right = np.einsum("abwghz,wzxy->abxghy", operator_right, quantum_channel)
        if site == site_1 or site == site_2:
            operator_right = np.einsum("abwghz,xw,yz ->abxghy", operator_right, operator, operator.conjugate())
        if site in quantum_channel_list:
            operator_right = np.einsum("abwghz,xywz->abxghy", operator_right, quantum_channel.conjugate())
        operator_right = np.einsum("abxghy,yea->ebxgh",operator_right, state[site])
        operator_right = np.einsum("ebxgh,xfb->efgh", operator_right, state[site].conjugate())

        operator_right_0 = np.einsum("abcd,xgc->abxgd", operator_right_0, state[site])
        operator_right_0 = np.einsum("abxgd,yhd->abxghy", operator_right_0, state[site].conjugate())
        if site in quantum_channel_list:
            operator_right_0 = np.einsum("abwghz,wzxy->abxghy", operator_right_0, quantum_channel)
            operator_right_0 = np.einsum("abwghz,xywz->abxghy", operator_right_0, quantum_channel.conjugate())
        operator_right_0 = np.einsum("abxghy,yea->ebxgh",operator_right_0, state[site])
        operator_right_0 = np.einsum("ebxgh,xfb->efgh", operator_right_0, state[site].conjugate())

        norm = np.linalg.norm(operator_right_0)
        operator_right, operator_right_0 = operator_right / norm, operator_right_0 / norm

    corr = operator_right[0, 0, 0, 0] / operator_right_0[0, 0, 0, 0]
    return corr


def renyi_2_susceptibility(state, site_list, operator, quantum_channel, quantum_channel_list):
    L = len(state)
    operator_right = np.zeros([1, 1, 2, 1, 1])
    operator_right[0, 0, 1, 0, 0] = 1
    operator_right_2 = np.zeros([1, 1, 2, 2, 1, 1])
    operator_right_2[0, 0, 1, 1, 0, 0] = 1

    s = operator.shape[0]
    m_tensor = np.zeros([s, s, s, s, 2, 2])
    m_tensor[:, :, :, :, 0, 0] = np.einsum("xw,yz->wzxy", np.eye(2), np.eye(2))
    m_tensor[:, :, :, :, 1, 1] = np.einsum("xw,yz->wzxy", np.eye(2), np.eye(2))
    m_tensor[:, :, :, :, 0, 1] = np.einsum("xw,yz->wzxy", operator, operator.conj())

    operator_right_0 = np.ones([1, 1, 1, 1])
    norm_list = np.zeros(L)
    for site in range(L-1, -1, -1):
        operator_right_0 = np.einsum("abcd,xgc->abxgd", operator_right_0, state[site])
        operator_right_0 = np.einsum("abxgd,yhd->abxghy", operator_right_0, state[site].conjugate())
        if site in quantum_channel_list:
            operator_right_0 = np.einsum("abwghz,wzxy->abxghy", operator_right_0, quantum_channel)
            operator_right_0 = np.einsum("abwghz,xywz->abxghy", operator_right_0, quantum_channel.conjugate())
        operator_right_0 = np.einsum("abxghy,yea->ebxgh", operator_right_0, state[site])
        operator_right_0 = np.einsum("ebxgh,xfb->efgh", operator_right_0, state[site].conjugate())

        norm_list[site] = np.linalg.norm(operator_right_0)
        operator_right_0 = operator_right_0 / norm_list[site]

    for site in range(L - 1, -1, -1):
        operator_right = np.einsum("abicd,xgc->abixgd", operator_right, state[site])
        operator_right = np.einsum("abixgd,yhd->abixghy", operator_right, state[site].conjugate())
        if site in quantum_channel_list:
            operator_right = np.einsum("abiwghz,wzxy->abixghy", operator_right, quantum_channel)
        if site in site_list:
            operator_right = np.einsum("abiwghz,wzxyji->abjxghy", operator_right, m_tensor)
        if site in quantum_channel_list:
            operator_right = np.einsum("abjwghz,xywz->abjxghy", operator_right, quantum_channel.conjugate())
        operator_right = np.einsum("abjxghy,yea->ebjxgh", operator_right, state[site])
        operator_right = np.einsum("ebjxgh,xfb->efjgh", operator_right, state[site].conjugate())

        operator_right = operator_right / norm_list[site]

    for site in range(L - 1, -1, -1):
        operator_right_2 = np.einsum("abikcd,xgc->abikxgd", operator_right_2, state[site])
        operator_right_2 = np.einsum("abikxgd,yhd->abikxghy", operator_right_2, state[site].conjugate())
        if site in quantum_channel_list:
            operator_right_2 = np.einsum("abikwghz,wzxy->abikxghy", operator_right_2, quantum_channel)
        if site in site_list:
            operator_right_2 = np.einsum("abikwghz,wzxyji->abjkxghy", operator_right_2, m_tensor)
            operator_right_2 = np.einsum("abjkwghz,wzxylk->abjlxghy", operator_right_2, m_tensor)
        if site in quantum_channel_list:
            operator_right_2 = np.einsum("abjlwghz,xywz->abjlxghy", operator_right_2, quantum_channel.conjugate())
        operator_right_2 = np.einsum("abjlxghy,yea->ebjlxgh", operator_right_2, state[site])
        operator_right_2 = np.einsum("ebjlxgh,xfb->efjlgh", operator_right_2, state[site].conjugate())

        operator_right_2 = operator_right_2 / norm_list[site]

    corr = operator_right[0, 0, 0, 0, 0] / operator_right_0[0, 0, 0, 0]
    corr_2 = operator_right_2[0, 0, 0, 0, 0, 0] / operator_right_0[0, 0, 0, 0]

    sus = (corr_2 - corr ** 2) / len(site_list)
    return sus


def check_renyi_2_susceptibility(state, site_list, operator, quantum_channel, quantum_channel_list):
    m = 0
    for site in site_list:
        m += renyi_2_correlator_torch(state, site, -2, operator, quantum_channel, quantum_channel_list)
    m = m

    m_2 = 0
    for site_1 in site_list:
        for site_2 in site_list:
            if site_2 != site_1:
                m_2 += renyi_2_correlator_torch(state, site_1, site_2, operator, quantum_channel, quantum_channel_list)
    m_2 = m_2 + len(site_list)
    m_2 = m_2
    return (m_2 - m ** 2) / len(site_list)


def check_renyi_2_correlator(state, L, site_1, site_2, operator):
    id = np.array([[1, 0], [0, 1]])
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])

    quantum_channel = np.kron(id, id) + np.kron(sx, sx)
    for site in range(L):
        n1, n2, n3 = state[site].shape
        state[site] = np.tensordot(state[site], state[site].conj(), axes=0).transpose(0, 3, 1, 4, 2, 5).reshape(n1 ** 2, n2 ** 2, n3 ** 2)
    for site in range(1, L - 1, 2):
        state[site] = np.einsum("abc,da->dbc", state[site], quantum_channel)

    my_mps = ZXZ_chain({'J': 1, 'h': 1, 'L': L, "cutoff_s": None, "cutoff_n": None, "n_sweep": None})
    my_mps.state = state
    corr = my_mps.measure_site_list([0, L-2], [np.kron(sz, sz), np.kron(sz, sz)])

    norm = my_mps.norm()
    print(corr, norm)
    print("corr:", corr / norm)

    return corr / norm


def _compute_one_correlation(i_h, model_type, pre_file_name):
    set_single_thread_env()

    with shelve.open(pre_file_name + str(i_h)) as db:
    # with shelve.open(pre_file_name + str(0)) as db:  # import dmrg result only from the first file: _0, output to many files
        state = db["state"]
        parameters = db["parameters"]

    # transpose each tensor
    for i in range(len(state)):
        state[i] = state[i].transpose(1, 0, 2)

    # initialize, in case it is not calculated
    string_order_parameter, corr, type_2_strange_corr = 0, 0, 0

    # string order parameter
    if model_type == "zxz":
        my_mps = ZXZ_chain({'J': 1, 'h': parameters['h'], 'L': len(state),
                            "cutoff_s": None, "cutoff_n": None, "n_sweep": None})
        my_mps.state = state

        id = np.array([[1, 0], [0, 1]])
        sx = np.array([[0, 1], [1, 0]])
        sz = np.array([[1, 0], [0, -1]])

        site_list = [my_mps.L//8 * 2] + [i for i in range(my_mps.L//8 * 2 + 1, my_mps.L // 8 * 6, 2)] + [my_mps.L // 8 * 6]
        operator_list = [sz] + [sx for _ in range(my_mps.L // 8 * 2)] + [sz]
    elif model_type == "spin_1":
        my_mps = Spin_1_Heisenberg_chain({'J': 1, 'h': parameters['h'], 'L': len(state),
                            "cutoff_s": None, "cutoff_n": None, "n_sweep": None})
        my_mps.state = state

        id = np.eye(3)
        sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        exp_sz = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
        exp_sx = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        site_list = [i for i in range(my_mps.L // 4, my_mps.L // 4 + my_mps.L // 2)]
        operator_list = [sx] + [exp_sx for _ in range(my_mps.L // 2 - 2)] + [sx]
    elif model_type == "zxz_gapless":
        my_mps = ZXZ_gapless_chain({'J': 1, 'h': parameters['h'], 'K': 1, 'L': len(state),
                            "cutoff_s": None, "cutoff_n": None, "n_sweep": None})
        my_mps.state = state
        id = np.array([[1, 0], [0, 1]])
        sx = np.array([[0, 1], [1, 0]])
        sz = np.array([[1, 0], [0, -1]])
        x0 = my_mps.L // 8 * 2 + 1
        site_list = [x0] + [i for i in range(x0 + 1, my_mps.L - x0, 2)] + [my_mps.L - x0]
        operator_list = [sz] + [sx for _ in range(my_mps.L // 2 - x0)] + [sz]
    else:
        raise ValueError("model_type is wrong")

    string_order_parameter = my_mps.measure_site_list(site_list, operator_list)

    # renyi-2 correlator
    if model_type == "zxz":
        p = 1 / 2
        quantum_channel = (1 - p) * np.kron(id, id) + p * np.kron(sx, sx)
        quantum_channel = quantum_channel.reshape(2, 2, 2, 2)
        quantum_channel_list = [i for i in range(1, my_mps.L, 2)]

        # corr = renyi_2_correlator(state, my_mps.L//8 * 2, my_mps.L // 8 * 6, sz, quantum_channel, quantum_channel_list)
        corr = renyi_2_correlator_torch(state, my_mps.L // 8 * 2, my_mps.L // 8 * 6, sz, quantum_channel, quantum_channel_list)
        type_2_strange_corr = type_2_strange_correlator(state, my_mps.L // 8 * 2 + 1, my_mps.L // 8 * 2 + my_mps.L // 8 * 4 + 1, sz, quantum_channel, quantum_channel_list)
    elif model_type == "spin_1":
        p = 1 / 2
        quantum_channel = (1-p) * np.kron(id, id) + p * np.kron(exp_sx, exp_sx)
        quantum_channel = quantum_channel.reshape(3, 3, 3, 3)
        quantum_channel_list = [i for i in range(0, my_mps.L)]

        #corr = renyi_2_correlator(state, my_mps.L // 4, my_mps.L // 4 + my_mps.L // 2, sx, quantum_channel, quantum_channel_list)
        corr = renyi_2_correlator_torch(state, my_mps.L // 4, my_mps.L // 4 + my_mps.L // 2, sx, quantum_channel, quantum_channel_list)
        type_2_strange_corr = type_2_strange_correlator(state, my_mps.L // 4, my_mps.L // 4 + my_mps.L // 2, sx, quantum_channel, quantum_channel_list)
    elif model_type == "zxz_gapless":
        p = 0.5  # change to i_h dependent if want to study different p
        quantum_channel = (1 - p) * np.kron(id, id) + p * np.kron(sx, sx)
        quantum_channel = quantum_channel.reshape(2, 2, 2, 2)
        quantum_channel_list = [2 * i for i in range(0, my_mps.L // 2)]
        sus_site_list = [2 * i + 1 for i in range(my_mps.L // 2)]

        ## corr = renyi_2_correlator(state, x0, my_mps.L - x0, sz, quantum_channel, quantum_channel_list)
        corr = renyi_2_correlator_torch(state, x0, my_mps.L - x0, sz, quantum_channel, quantum_channel_list)
        ## corr = renyi_2_correlator_torch(state, x0, x0 + i_h * 100, sz, quantum_channel, quantum_channel_list)
        ## corr = renyi_2_susceptibility(state, sus_site_list, sz, quantum_channel, quantum_channel_list)
        #corr = renyi_2_susceptibility_torch(state, sus_site_list, sz, quantum_channel, quantum_channel_list)
    else:
        raise ValueError("model_type is wrong")


    # strange correlator
    L = len(state)
    if model_type == "zxz":
        sz = np.array([[1, 0], [0, -1]])
        strange_corr = strange_correlator(state, L // 8 * 2, L // 8 * 2 + L // 8 * 4, sz)
    else:
        strange_corr = 0

    # return everything needed (keep order by i_h)
    return {
        "i_h": i_h,
        "h": parameters['h'],
        "string": string_order_parameter,
        "corr": corr,
        "strange_corr": strange_corr,
        "type_2_strange_corr": type_2_strange_corr,
    }


def _compute_one_dmrg(i_h, h, model_type):
    set_single_thread_env()

    if model_type == "zxz":
        result = run_dmrg_zxz(L=1000, J=1.0, h=h, chi_max=100, sweeps=100)
    elif model_type == "spin_1":
        result = run_dmrg_spin1_heisenberg(L=2000, J=1.0, h=h, chi_max=100, sweeps=100)
    elif model_type == "zxz_gapless":
        result = run_dmrg_zxz_gapless(L=4000, J=1.0, h=h, K=1.0, chi_max=80, sweeps=100)
    else:
        raise ValueError("model_type is wrong")

    return {
        "i_h": i_h,
        "state": result["state"],
        "parameters": result["parameters"]
    }


def main():
    N_h = 2
    h_list = [0.8 + i_h * 0.05 for i_h in range(N_h)]
    model_type = "zxz" # "spin_1", "zxz", "zxz_gapless"
    pre_file_name = "data/" + model_type + "_model/tenpy_result_"

    max_workers = 8  # or an int, e.g., 8

    start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = ex.map(_compute_one_dmrg, range(N_h), h_list, [model_type] * N_h)
    end = time.time()
    print("time_dmrg:", end - start)

    for res in results:
        with shelve.open(pre_file_name + str(res["i_h"])) as db:
            db["state"] = res["state"]
            db["parameters"] = res["parameters"]

    start = time.time()
    h_list_from_data = []
    corr_list = []
    order_parameter_list = []
    strange_corr_list = []
    type_2_corr_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # map preserves input order, so results align with i_h = 0..N_h-1
        results = list(ex.map(_compute_one_correlation, range(N_h), [model_type] * N_h, [pre_file_name] * N_h))
    end = time.time()
    print("time_corr:", end - start)

    # consume results
    for res in results:
        print(res["i_h"], res["h"])
        print("string:", res["string"])
        print("renyi_2_correlator:", res["corr"])
        print("strange_correlator:", res["strange_corr"])
        print("type_2_strange_correlator:", res["type_2_strange_corr"])

        h_list_from_data.append(res["h"])
        order_parameter_list.append(res["string"])
        corr_list.append(res["corr"])
        strange_corr_list.append(res["strange_corr"])
        type_2_corr_list.append(res["type_2_strange_corr"])

    with shelve.open(re.sub(r'([/\\])tenpy_result_', r'\1result', pre_file_name)) as db:
        db["h_list"] = h_list_from_data
        db["order_parameter_list"] = order_parameter_list
        db["corr_list"] = corr_list
    with shelve.open(re.sub(r'([/\\])tenpy_result_', r'\1result_strange_correlator', pre_file_name)) as db:
        db["strange_corr_list"] = strange_corr_list
        db["type_2_corr_list"] = type_2_corr_list


def main_multiplejobs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    parser.add_argument("--h_0", type=float)
    parser.add_argument("--h_step", type=float)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--run_mode", type=str)
    args = parser.parse_args()

    print(f"Running ID={args.id}, h_0={args.h_0}, h_step={args.h_step}, folder={args.folder}, mode={args.mode}, run_mode{args.run_mode}")

    i_h = args.id
    h = args.h_0 + i_h * args.h_step
    model_type = args.mode # "spin_1", "zxz", "zxz_gapless"
    
    out_dir = os.path.join("data", model_type + "_model", args.folder)
    os.makedirs(out_dir, exist_ok=True)
    pre_file_name = os.path.join(out_dir, "tenpy_result_")

    if args.run_mode == "dmrg":
        start = time.time()
        result = _compute_one_dmrg(i_h, h, model_type)
        end = time.time()
        print("time_dmrg:", end - start)

        with shelve.open(pre_file_name + str(i_h)) as db:
            db["h"] = h
            db["state"] = result["state"]
            db["parameters"] = result["parameters"]

    if args.run_mode == "correlator":
        start = time.time()
        result = _compute_one_correlation(i_h, model_type, pre_file_name)
        end = time.time()
        print("time_corr:", end - start)

        print("corr_result:", result)

        with shelve.open(re.sub(r'([/\\])tenpy_result_', r'\1result', pre_file_name) + "_" + str(i_h)) as db:
            db["h"] = result["h"]
            db["order_parameter"] = result["string"]
            db["corr"] = result["corr"]
        with shelve.open(re.sub(r'([/\\])tenpy_result_', r'\1result_strange_correlator', pre_file_name) + "_" + str(i_h)) as db:
            db["strange_corr"] = result["strange_corr"]
            db["type_2_corr"] = result["type_2_strange_corr"]


if __name__ == "__main__":
    main()
    # main_multiplejobs()




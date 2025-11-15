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
import matplotlib.pyplot as plt

# # change file name and save file name
# # zxz
# filename = "data/zxz_model/tenpy_result_1"
# id = np.array([[1, 0], [0, 1]])
# sx = np.array([[0, 1], [1, 0]])
# sz = np.array([[1, 0], [0, -1]])
#
# # spin_1
# # filename = "data/spin_1_model/tenpy_result_1"
# # id = np.eye(3)
# # sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
# # exp_sz = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
# # sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
# # exp_sx = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
#
# with shelve.open(filename) as db:
#     state = db["state"]
#     parameters = db["parameters"]
#
# for i in range(len(state)):
#     state[i] = state[i].transpose(1, 0, 2)
#
# L = len(state)
# D = state[0].shape[0]
# L_mid = L // 2
# l = 3
#
# operator_left = np.ones([1, 1])
# operator_right = np.ones([1, 1])
# for i in range(L_mid - l):
#     operator_left = np.einsum("ab,xac->xcb", operator_left, state[i])
#     operator_left = np.einsum("xcb,xbd->cd", operator_left, state[i].conj())
#
#     operator_right = np.einsum("cd,xac->xad", operator_right, state[L - 1 - i])
#     operator_right = np.einsum("xad,xbd->ab", operator_right, state[L - 1 - i].conj())
#
# N_p = 3
# negativity_list = []
# p_list = [i_p * 0.5 / (N_p - 1) for i_p in range(N_p)]
# for p in p_list:
#     # zxz
#     quantum_channel = (1 - p) * np.kron(id, id) + p * np.kron(sx, sx)
#     quantum_channel = quantum_channel.reshape(2, 2, 2, 2)
#     quantum_channel_list = [i for i in range(1, L, 2)]
#     # spin_1
#     # quantum_channel = (1-p) * np.kron(id, id) + p * np.kron(exp_sx, exp_sx)
#     # quantum_channel = quantum_channel.reshape(3, 3, 3, 3)
#     # quantum_channel_list = [i for i in range(0, L)]
#
#     density_matrix = np.einsum("xy,ab->xyab", np.ones([1, 1]), operator_left)
#     for i in range(L_mid - l, L_mid +l):
#         density_matrix = np.einsum("xyab,wac->xywcb", density_matrix, state[i])
#         density_matrix = np.einsum("xywcb,zbd->xywzcd", density_matrix, state[i].conj())
#         if i in quantum_channel_list:
#             density_matrix = np.einsum("xystcd,stwz->xywzcd", density_matrix, quantum_channel)
#         density_matrix = density_matrix.transpose(0, 2, 1, 3, 4, 5)
#         D0, D1, D2, D3, D4, D5 = density_matrix.shape
#         density_matrix = density_matrix.reshape(D0*D1, D2*D3, D4, D5)
#     density_matrix = np.einsum("xyab,ab->xy", density_matrix, operator_right)
#
#     density_matrix = density_matrix.reshape(D ** l, D ** l, D ** l, D ** l)
#     density_matrix = density_matrix.transpose(0, 3, 2, 1)
#     density_matrix = density_matrix.reshape(D ** (2*l), D ** (2*l))
#
#     eigvals, eigvecs = np.linalg.eig(density_matrix)
#     neg_eigs = eigvals[eigvals < 0]
#     negativity = - np.real(np.sum(neg_eigs))
#     print(p, negativity)
#     negativity_list.append(negativity)
#
# print(negativity_list)
# # # zxz / spin_1
# # with shelve.open("data/negativity_data/spin_1_1") as db:
# #     db["p_list"] = p_list
# #     db["negativity"] = negativity_list





# # plot figures
# with shelve.open("data/negativity_data/zxz_0") as db:
#     p_list = db["p_list"]
#     negativity_list = db["negativity"]
# plt.plot(p_list, negativity_list, "o-", linewidth=0.8, markersize=2)
# with shelve.open("data/negativity_data/zxz_1") as db:
#     p_list = db["p_list"]
#     negativity_list = db["negativity"]
# plt.plot(p_list, negativity_list, "o-", linewidth=0.8, markersize=2)
# with shelve.open("data/negativity_data/spin_1_0") as db:
#     p_list = db["p_list"]
#     negativity_list = db["negativity"]
# plt.plot(p_list, negativity_list, "o-", linewidth=0.8, markersize=2)
# with shelve.open("data/negativity_data/spin_1_1") as db:
#     p_list = db["p_list"]
#     negativity_list = db["negativity"]
# plt.plot(p_list, negativity_list, "o-", linewidth=0.8, markersize=2)
#
# plt.xlabel("p")
# plt.ylabel("negativity")
# plt.savefig("negativity.pdf")


def entropy(density_matrix):
    eigs, _ = np.linalg.eig(density_matrix)
    eigs = eigs[eigs > 10 ** (-10)]
    entropy = - eigs @ np.log(eigs)
    return entropy

# check negativity zxz model
L = 6
h= 0.5
H = np.zeros((2 ** L, 2 ** L))
id = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
zxz = np.kron(sz, np.kron(sx, sz))
for i in range(1, L-1):
    H -= np.kron(np.eye(2 ** (i - 1)), np.kron(zxz, np.eye(2 ** (L - i - 2))))
for i in range(L):
    H -= h * np.kron(np.eye(2 ** i), np.kron(sx, np.eye(2 ** (L - i - 1))))

x_even = np.kron(np.kron(np.kron(sx, np.eye(2)), np.kron(sx, np.eye(2))), np.kron(sx, np.eye(2)))
x_odd = np.kron(np.kron(np.kron(np.eye(2), sx), np.kron(np.eye(2), sx)), np.kron(np.eye(2), sx))
H -= x_even + x_odd
# H -= np.kron(np.kron(sz, np.eye(2 ** (L - 3))), np.kron(sz, sx))
# H -= np.kron(np.kron(sx, sz), np.kron(np.eye(2 ** (L - 3)), sz))

eigvals, eigvecs = np.linalg.eigh(H)
psi = eigvecs[:, 0]
print(eigvals)
density_matrix = np.outer(psi.conj(), psi)

# projection
# projected_spins = [None, 0, None, 0, None, 0]
p = 0.5
for i in range(1, L, 2):
    x = np.kron(np.eye(2 ** i), np.kron(sx, np.eye(2 ** (L - i - 1))))
    density_matrix = (1-p) * density_matrix + p * x @ density_matrix @ x
    # P_plus = np.kron(np.eye(2 ** i), np.kron((sx + id)/2, np.eye(2 ** (L - i - 1))))
    # P_minus = np.kron(np.eye(2 ** i), np.kron((id - sx) / 2, np.eye(2 ** (L - i - 1))))
    # if projected_spins[i] == 0:
    #     density_matrix = P_plus @ density_matrix @ P_plus
    # if projected_spins[i] == 1:
    #     density_matrix = P_minus @ density_matrix @ P_minus


density_matrix_half_transpose = density_matrix.reshape(2 ** (L // 2), 2 ** (L // 2), 2 ** (L // 2), 2 ** (L // 2)).transpose(0, 3, 2, 1).reshape(2 ** L, 2 ** L)
eigvals, _ = np.linalg.eigh(density_matrix_half_transpose)
negativity = -np.sum(eigvals[eigvals < 0])
#print(np.sort(eigvals))
print("negativity:", negativity)


density_matrix_ac = density_matrix.reshape(16, 2, 2, 16, 2, 2)
density_matrix_ac = np.einsum("abcdbe->acde", density_matrix_ac).reshape(32, 32)
density_matrix_bc = density_matrix.reshape(4, 2, 8, 4, 2, 8)
density_matrix_bc = np.einsum("abcdbe->acde", density_matrix_bc).reshape(32, 32)
density_matrix_b = density_matrix.reshape(4, 2, 2, 2, 2, 4, 2, 2, 2, 2)
density_matrix_b = np.einsum("xaybzmanbp->xyzmnp", density_matrix_b).reshape(16, 16)

s_abc = entropy(density_matrix)
print("abc", s_abc)
s_ac = entropy(density_matrix_ac)
print("ac", s_ac)
s_bc = entropy(density_matrix_bc)
print("bc", s_bc)
s_b = entropy(density_matrix_b)
print("b", s_b)
cmi = s_ac + s_bc - s_b - s_abc
print("cmi:", cmi)

































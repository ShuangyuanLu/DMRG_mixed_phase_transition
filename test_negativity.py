import numpy as np


def entropy(density_matrix):
    eigs, _ = np.linalg.eig(density_matrix)
    eigs = eigs[eigs > 10 ** (-10)]
    entropy = - eigs @ np.log(eigs)
    return entropy

# check negativity zxz model
L = 12
h= 0.5
H = np.zeros((2 ** L, 2 ** L))
id = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
zxz = np.kron(sz, np.kron(sx, sz))
np.random.seed(0)
for i in range(1, L-1):
    # H -= (1 + np.random.rand()) * np.kron(np.eye(2 ** (i - 1)), np.kron(zxz, np.eye(2 ** (L - i - 2))))
    H -= np.kron(np.eye(2 ** (i - 1)), np.kron(zxz, np.eye(2 ** (L - i - 2))))
for i in range(L):
    # H -= h * (1 + 0.1 * np.random.rand()) * np.kron(np.eye(2 ** i), np.kron(sx, np.eye(2 ** (L - i - 1))))
    H -= h * np.kron(np.eye(2 ** i), np.kron(sx, np.eye(2 ** (L - i - 1))))
# for i in range(L-1):
#     H -= 0.1 * np.kron(np.eye(2 ** i), np.kron(np.kron(sx, sx), np.eye(2 ** (L - i - 2))))
# for i in range(L-2):
#     H -= 0.1 * np.kron(np.eye(2 ** i), np.kron(np.kron(sz, np.kron(np.eye(2), sz)), np.eye(2 ** (L - i - 3))))


x_even = np.eye(1)
x_odd = np.eye(1)
for i in range(L // 2):
    x_even = np.kron(np.kron(sx, np.eye(2)), x_even)
    x_odd = np.kron(np.kron(np.eye(2), sx), x_odd)
H -= x_even + x_odd
H -= np.kron(np.kron(sz, np.eye(2 ** (L - 3))), np.kron(sz, sx))
H -= np.kron(np.kron(sx, sz), np.kron(np.eye(2 ** (L - 3)), sz))

eigvals, eigvecs = np.linalg.eigh(H)
psi = eigvecs[:, 0]
#projected_spins = [None, 1, None, 0, None, 1, None, 0]

def f(projected_spins, psi):
    for i in range(1, L, 2):
        P_plus = np.kron(np.eye(2 ** i), np.kron((sx + id)/2, np.eye(2 ** (L - i - 1))))
        P_minus = np.kron(np.eye(2 ** i), np.kron((id - sx) / 2, np.eye(2 ** (L - i - 1))))
        if projected_spins[i] == 0:
            psi = P_plus @ psi
        if projected_spins[i] == 1:
            psi = P_minus @ psi

    psi = psi.reshape([2 for _ in range(L)])
    psi = psi[tuple(slice(None) if s is None else int(s) for s in projected_spins)] * (np.sqrt(2) ** (L // 2))
    weight = np.linalg.norm(psi)
    psi = psi / weight

    # # psi = psi.reshape(2 ** (L // 4), 2 ** (L // 4))
    # psi = psi.reshape(4, 8)
    # U, S, V = np.linalg.svd(psi, full_matrices=False)
    # print(S)
    return weight

x0 = f([None, 0, None, 0, None, 0, None, 0, None, 0, None, 0], psi)
x1 = f([None, 0, None, 1, None, 0, None, 1, None, 0, None, 0], psi)
x2 = f([None, 0, None, 0, None, 1, None, 1, None, 0, None, 0], psi)
x3 = f([None, 1, None, 0, None, 0, None, 1, None, 0, None, 0], psi)
x4 = f([None, 1, None, 1, None, 1, None, 1, None, 1, None, 1], psi)

print(x0)
print(x1)
print(x2)
print(x3)
print(x4)

print("-" * 80)
print((x3 / x0) ** 2 * x0)

# density_matrix = np.outer(psi.conj(), psi)

# projection
# projected_spins = [None, 0, None, 0, None, 0, None, 0]
# p = 0.5
# for i in range(1, L, 2):
#     P_plus = np.kron(np.eye(2 ** i), np.kron((sx + id)/2, np.eye(2 ** (L - i - 1))))
#     P_minus = np.kron(np.eye(2 ** i), np.kron((id - sx) / 2, np.eye(2 ** (L - i - 1))))
#     if projected_spins[i] == 0:
#         density_matrix = P_plus @ density_matrix @ P_plus
#     if projected_spins[i] == 1:
#         density_matrix = P_minus @ density_matrix @ P_minus
#
# weight = np.trace(density_matrix)
# print("weight:", weight)
# density_matrix = density_matrix
#
# density_matrix_half_transpose = density_matrix.reshape(2 ** (L // 2), 2 ** (L // 2), 2 ** (L // 2), 2 ** (L // 2)).transpose(0, 3, 2, 1).reshape(2 ** L, 2 ** L)
#
# # eigvals, eigvecs = np.linalg.eigh(density_matrix_half_transpose)
# # print(eigvals[: 10]/ weight)
# # # negativity = -np.sum(eigvals[eigvals < 0])
# # # #print(np.sort(eigvals))
# # # print("negativity:", negativity/weight)

#
# density_matrix_ac = density_matrix.reshape(16, 2, 2, 16, 2, 2)
# density_matrix_ac = np.einsum("abcdbe->acde", density_matrix_ac).reshape(32, 32)
# density_matrix_bc = density_matrix.reshape(4, 2, 8, 4, 2, 8)
# density_matrix_bc = np.einsum("abcdbe->acde", density_matrix_bc).reshape(32, 32)
# density_matrix_b = density_matrix.reshape(4, 2, 2, 2, 2, 4, 2, 2, 2, 2)
# density_matrix_b = np.einsum("xaybzmanbp->xyzmnp", density_matrix_b).reshape(16, 16)
#
# s_abc = entropy(density_matrix)
# s_ac = entropy(density_matrix_ac)
# s_bc = entropy(density_matrix_bc)
# s_b = entropy(density_matrix_b)
# cmi = s_ac + s_bc - s_b - s_abc
# print("cmi:", cmi / weight)

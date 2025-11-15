import numpy as np
import scipy
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh, eig
from partial_trace import partial_trace
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
import math
from functools import partial
from scipy.sparse import eye, kron, csr_matrix, csc_matrix, coo_matrix


def gate_map():
    L = 6
    id = np.eye(2)
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    sp = np.array([[0, 1], [0, 0]])
    sm = np.array([[0, 0], [1, 0]])
    Pp = np.array([[1, 0], [0, 0]])
    Pm = np.array([[0, 0], [0, 1]])
    n_digits = 4

    zz = np.kron(sz, sz)
    rho_env_J = (np.kron(id, id) + zz) / 4
    H_J = np.kron(np.kron(Pp, sm), np.kron(Pp, sp)) + np.kron(np.kron(Pp, sp), np.kron(Pp, sm)) + np.kron(np.kron(sm, Pp), np.kron(sp, Pp)) + np.kron(np.kron(sp, Pp), np.kron(sm, Pp)) \
      + np.kron(np.kron(Pm, sm), np.kron(Pm, sp)) + np.kron(np.kron(Pm, sp), np.kron(Pm, sm)) + np.kron(np.kron(sm, Pm), np.kron(sp, Pm)) + np.kron(np.kron(sp, Pm), np.kron(sm, Pm))

    rho_env_h = (id + sx) / 2
    H_h = np.kron(sz, sz) + np.kron(sy, sy)

    #rho_env = (np.kron(id, id) + np.kron(id, sz)) / 4
    #H = np.kron(np.kron(id, sx), np.kron(id, sx)) + np.kron(np.kron(id, sy), np.kron(id, sy))
    #dim = 4

    U_h = scipy.linalg.expm(1j * 0.4 * H_h)     # pi/4
    U_J = scipy.linalg.expm(1j * math.pi / 2 * H_J)
    print(U_h)

    # theta = math.pi / 4
    # U = (np.eye(4) + np.kron(sz, sz)) / 2 + np.matmul((np.eye(4) - np.kron(sz, sz)), (np.eye(4) * math.cos(theta) + np.kron(sx, sx) * math.sin(theta))) / 2

    def evolution(rho_sys_input, U, rho_env, dim):
        rho_sys_input = np.reshape(rho_sys_input, [dim, dim])
        rho_sys_output = partial_trace(np.matmul(np.matmul(U, np.kron(rho_env, rho_sys_input)), (U.conj().transpose())), 1, [dim, dim])
        rho_sys_output = np.reshape(rho_sys_output, [dim ** 2])
        return rho_sys_output

    evolution_operator_J = LinearOperator((4 ** 2, 4 ** 2), matvec=partial(evolution, dim=4, U=U_J, rho_env=rho_env_J))
    evolution_operator_h = LinearOperator((2 ** 2, 2 ** 2), matvec=partial(evolution, dim=2, U=U_h, rho_env=rho_env_h))

    #eigenvalues_k, eigenstates_k = eigs(evolution_operator, k=2, which='LM')
    #rho_sys_ness = np.reshape(eigenstates_k[:, 0], [dim, dim])
    #rho_sys_ness = rho_sys_ness / np.trace(rho_sys_ness)
    #first_mode = np.reshape(eigenstates_k[:, 1], [dim, dim])

    evolution_matrix_J = linear_operator_to_matrix(evolution_operator_J, 4 ** 2)
    evolution_matrix_h = linear_operator_to_matrix(evolution_operator_h, 2 ** 2)
    print(evolution_matrix_h)

    evolution_matrix_J = evolution_matrix_J.reshape((2, 2, 2, 2, 2, 2, 2, 2))
    evolution_matrix_J = evolution_matrix_J.transpose((0, 2, 1, 3, 4, 6, 5, 7))
    evolution_matrix_J = evolution_matrix_J.reshape((4 ** 2, 4 ** 2))

    evolution_matrix_J_sparse = coo_matrix(evolution_matrix_J)
    #print("evolution_matrix_J:")
    #print(evolution_matrix_J_sparse)
    evolution_matrix_h[np.abs(evolution_matrix_h) < 10 ** (-15)] = 0
    #print("evolution_matrix_H:")
    #print(evolution_matrix_h)
    evolution_matrix_h_sparse = coo_matrix(evolution_matrix_h)

    #print("error:", np.sum(np.abs(evolution_matrix_h_sparse - np.kron(sx, sx).dot(evolution_matrix_h_sparse.dot(np.kron(sx, sx))))))
    #print("error:", np.sum(np.abs(evolution_matrix_J_sparse - np.kron(np.kron(sx, sx), np.kron(sx, sx)).dot(evolution_matrix_J_sparse.dot(np.kron(np.kron(sx, sx),np.kron(sx, sx)))))))

    #evolution_matrix_J_1 = np.matmul(np.kron(evolution_matrix_h, evolution_matrix_h), evolution_matrix_J)
    #evolution_matrix = np.kron(np.eye(4), np.kron(np.kron(evolution_matrix_J, evolution_matrix_J), np.eye(4)))
    #evolution_matrix = np.matmul(np.kron(evolution_matrix_J_1, np.kron(evolution_matrix_J_1, evolution_matrix_J_1)), evolution_matrix)

    def evolution_L(state_input):
        for i in range(L // 2 - 1):
            site = 2 * i + 1
            evolution_site = kron(eye(4 ** site), kron(evolution_matrix_J_sparse, eye(4 ** (L - site - 2))))
            state_input = evolution_site.dot(state_input)
        for i in range(L // 2):
            site = 2 * i
            evolution_site = kron(eye(4 ** site), kron(evolution_matrix_J_sparse, eye(4 ** (L - site - 2))))
            state_input = evolution_site.dot(state_input)
        for site in range(L):
            evolution_site = kron(eye(4 ** site), kron(evolution_matrix_h_sparse, eye(4 ** (L - site - 1))))
            state_input = evolution_site.dot(state_input)
        return state_input

    '''
    state_exact = [np.ones(1) for _ in range(4)]
    for i in range(L):
        for j in range(4):
            state_exact[j] = np.kron(state_exact[j], np.eye(4)[j, :])
    state_exact = np.transpose(np.array(state_exact))

    effective_evolution = evolution_L(state_exact)
    effective_evolution = np.transpose(effective_evolution).dot(effective_evolution)
    print("effective_evolution")
    print(np.round(effective_evolution, 16))
    '''

    evolution_L_operator = LinearOperator((4 ** L, 4 ** L), matvec=evolution_L)
    eigenvalues, eigenstates = eigs(evolution_L_operator, k=8, which='LM')
    #eigenvalues, eigenstates = eigs(evolution_matrix, k=8, which='LM')

    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenstates = eigenstates[:, sorted_indices]
    print(eigenvalues)
    #print("eigenstates:", eigenstates[:, 0: 5])
    #print("eigenvalues:", eigenvalues)
    #print("ness:")
    ness = eigenstates[:, 0].reshape([2 for _ in range(2 *L)])
    ness = ness.transpose([2 * i for i in range(L)] + [2 * i + 1 for i in range(L)])
    ness = ness.reshape((2 ** L, 2 ** L))
    #print("trace:", np.trace(ness))
    ness = ness / np.trace(ness)
    #print(np.round(ness, 8))

    for i in range(1, L):
        sx_i = np.kron(np.kron(np.eye(2 ** i), sx), np.eye(2 ** (L - 1 - i)))
        sz_i = np.kron(np.kron(np.eye(2), np.kron(np.eye(2 ** (i - 1)), sz)), np.eye(2 ** (L - 1 - i)))
        sz_corr = np.kron(np.kron(sz, np.kron(np.eye(2 ** (i - 1)), sz)), np.eye(2 ** (L - 1 - i)))
        print("sx:", i, np.trace(np.matmul(sx_i, ness)))
        print("sz:", i, np.trace(np.matmul(sz_i, ness)))
        print("sz_corr:", i, np.trace(np.matmul(sz_corr, ness)))

    #plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', marker='+')
    #plt.show()

    #print(rho_sys_ness)
    #print(first_mode)
    #print(eigenvalues)

    #weights, states = eigh(rho_sys_ness)
    #print(weights)
    #print(states)

    #return rho_sys_ness


def linear_operator_to_matrix(operator, n):
    basis_vectors = np.eye(n)  # Standard basis vectors (identity matrix)
    return np.column_stack([operator(basis_vectors[:, i]) for i in range(n)])

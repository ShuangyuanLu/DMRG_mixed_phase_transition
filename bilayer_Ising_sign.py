import numpy as np
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import eye, kron, csr_matrix, csc_matrix, coo_matrix
from scipy import sparse

L = 6
Id = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=float))
X = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
Z = sparse.csr_matrix(np.array([[1,  0], [0, -1]], dtype=float))

alpha =  2
beta = 3

np.set_printoptions(linewidth=200, threshold=np.inf)

def hamiltonian_map(state_input):
    state_output = np.zeros(state_input.shape)
    for i in range(L):
        X_i = kron(kron(eye(2 ** i), X), eye(2 ** (L - 1 - i)))
        state_output += - kron(X_i, eye(2 ** L)).dot(state_input)
        state_output += - kron(eye(2 ** L), X_i).dot(state_input)
        state_output += - alpha * kron(X_i, X_i).dot(state_input)
    for i in range(L):
        if i == L - 1:
            continue    # OBC
            #ZZ_i = kron(kron(Z, eye(2 ** (L - 2))), Z)   #PBC
        else:
            ZZ_i = kron(kron(eye(2 ** i), kron(Z, Z)), eye(2 ** (L - 2 - i)))
        state_output += - beta * kron(ZZ_i, eye(2 ** L)).dot(state_input)
        state_output += - beta * kron(eye(2 ** L), ZZ_i).dot(state_input)
        state_output += - (beta * alpha) * kron(ZZ_i, ZZ_i).dot(state_input)

    return state_output

hamiltonian_operator = LinearOperator(shape=(2 ** (2 * L), 2 ** (2 * L)), matvec=hamiltonian_map)
eigenvalue, eigenstate = eigsh(hamiltonian_operator, k=1, which='SA')

rho = eigenstate.reshape((2 ** L, 2 ** L))
#schmidt_eigenvalue, _ = eig(rho)
#print(schmidt_eigenvalue)

U, S, V = np.linalg.svd(rho, full_matrices=False)
sign = np.einsum('ij,ij->j', U, V.conj().T)
sign = sign * sign[0]
sign = sign[S > 10 ** (-10)]
print("diff_norm =", np.linalg.norm(rho - rho.conj().T, 'fro'))
print(eigenstate.shape)
#operator =
#order_parameter = eigenstate.conj().T.dot(operator.dot(eigenstate))
#print(S)
print(eigenvalue)
#print(S)
#print(U)
#print(V.conj().T)
print(sign)

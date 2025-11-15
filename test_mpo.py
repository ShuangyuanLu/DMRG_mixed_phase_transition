import numpy as np
import math
from scipy.linalg import eigh
from numpy.linalg import svd, norm
np.set_printoptions(linewidth=200)


def find_intertwiner(A, B, rtol=1e-10):
    """
    Find X (chi_B x chi_A) such that X A^s = B^s X for all s (complex allowed).

    Parameters
    ----------
    A, B : np.ndarray
        Shapes (d, chi_A, chi_A) and (d, chi_B, chi_B).
    rtol : float
        Relative tolerance to identify (near-)null singular values.

    Returns
    -------
    X : np.ndarray
        Intertwiner of shape (chi_B, chi_A), defined up to overall scale.
    info : dict
        Diagnostics: residual, nullity, singular_values, threshold, rankX.
    """
    d, chi_A1, chi_A2 = A.shape
    dB, chi_B1, chi_B2 = B.shape
    assert d == dB and chi_A1 == chi_A2 and chi_B1 == chi_B2, "shapes must be (d, chi, chi)"
    chi_A, chi_B = chi_A1, chi_B1

    I_A = np.eye(chi_A)
    I_B = np.eye(chi_B)

    # Build K vec(X)=0 by stacking over s
    blocks = []
    for s in range(d):
        Ks = np.kron(I_B, A[s].T) - np.kron(B[s], I_A)   # (chi_B*chi_A) x (chi_B*chi_A)
        blocks.append(Ks)
    K = np.vstack(blocks)

    # SVD to get nullspace / least-squares solution
    U, S, Vh = svd(K, full_matrices=False)
    thresh = max(rtol * S[0], rtol) if S.size else rtol
    nullity = int(np.sum(S <= thresh))
    x_vec = Vh[-1] if S.size else np.eye(chi_B, chi_A).ravel()  # fallback
    X = x_vec.reshape(chi_B, chi_A)

    # Normalize (scale is arbitrary)
    fn = norm(X)
    if fn > 0:
        X = X / fn

    # Diagnostics
    resid = 0.0
    for s in range(d):
        resid += norm(X @ A[s] - B[s] @ X, 'fro')**2
    resid = np.sqrt(resid)

    # rank of X (useful: isometry/surjection)
    rX = np.linalg.matrix_rank(X, tol=1e-12)

    info = {
        "residual": resid,
        "nullity": nullity,
        "singular_values": S,
        "threshold": thresh,
        "rankX": rX,
        "chi_A": chi_A,
        "chi_B": chi_B,
    }
    return X, info


def tensors_close(A, B, rtol=1e-8, atol=1e-10):
    return np.allclose(A, B, rtol=rtol, atol=atol)






id = np.array([[1, 0], [0, 1]], dtype=np.complex128)
sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

A = np.zeros((2, 2, 2), dtype=np.complex128)
B = np.zeros((2, 2, 2), dtype=np.complex128)
A[0, 0, :] = np.array([1, 0])
A[1, 1, :] = np.array([0, 1])
B[0, 0, :] = np.array([1, 1]) / math.sqrt(2)
B[1, 1, :] = np.array([1, 1]) / math.sqrt(2)
B[0, 1, :] = np.array([1, -1]) / math.sqrt(2)
B[1, 0, :] = np.array([1, -1]) / math.sqrt(2)

D = np.zeros((2, 2, 2, 2), dtype=np.complex128)
D[0, 0, :, :] = id + 1j * sx
D[0, 1, :, :] = 1j * sz + 1j * sy
D[1, 0, :, :] = sz - sy
D[1, 1, :, :] = 1j * id + sx
D = (1 - 1j) / math.sqrt(2) / 2  * D

DA = np.einsum("abcd,ef->aebfcd", D, id).reshape(4, 4, 2, 2)
DB = np.einsum("ab,efcd->aebfcd", id, D).reshape(4, 4, 2, 2)

DA_A = np.einsum("dfec,abc->dafbe", DA, A).reshape(8, 8, 2)
DB_B = np.einsum("dfec,abc->dafbe", DB, B).reshape(8, 8, 2)

DAB_AB = np.einsum("abc,bef->aecf", DA_A, DB_B).reshape(8, 8, 4)
BA = np.einsum("abc,bef->aecf", B, A).reshape(2, 2, 4)

BA_transfer_matrix = np.einsum("abc,dec->adbe", BA, BA.conj()).reshape(4, 4)
eigvals, eigvecs = np.linalg.eig(BA_transfer_matrix)
print(eigvals)
print(eigvecs)
BA_DAB_AB_transfer_matrix = np.einsum("abc,dec->adbe", BA, DAB_AB.conj()).reshape(16, 16)
eigvals, eigvecs = np.linalg.eig(BA_DAB_AB_transfer_matrix)
print(eigvals)
print(eigvecs[:, 2].reshape(2, 8))
print(eigvecs[:, 3].reshape(2, 8))

D_AB_AB_transfer_matrix = np.einsum("abc,dec->adbe", DAB_AB, DAB_AB.conj()).reshape(64, 64)
eigvals, eigvecs = np.linalg.eig(D_AB_AB_transfer_matrix)
print(np.sort(np.abs(eigvals)))

V = np.zeros((4, 8), dtype=np.complex128)
V[0, 0], V[0, 3], V[0, 5], V[0, 6] = 1, 1j, 1, 1j
V[1, 1], V[1, 2], V[1, 4], V[1, 7] = 1, 1j, 1, 1j
V[2, 1], V[2, 2], V[2, 4], V[2, 7] = 1, 1j, -1, -1j
V[3, 0], V[3, 3], V[3, 5], V[3, 6] = -1, -1j, 1, 1j
V = V[2:4, :]

# BA_double = np.einsum("de, abc->daebc", id, BA).reshape(4, 4, 4)
print(tensors_close(V.conj().transpose() @ BA[:, :, 0] @ V, DAB_AB[:, :, 0]))
print(V.conj().transpose() @ BA[:, :, 0] @ V)
print(DAB_AB[:, :, 2] * (1 + 1j) * 4 / math.sqrt(2))
print("-" * 200)
eigvals, eigvecs = np.linalg.eig(DAB_AB[:, :, 0] * (1 + 1j) * 4 / math.sqrt(2))
print(DAB_AB[:, :, 1]* (1 + 1j) * 4 / math.sqrt(2))
print(eigvals)
print(eigvecs)
#
# DAB_AB, BA = DAB_AB.transpose(2, 0, 1), BA_double.transpose(2, 0, 1)
# X, info = find_intertwiner(DAB_AB, BA)
# print(X.shape)
# print("Residual (should be ~0):", info["residual"])
# print("Nullity (injective case -> 1):", info["nullity"])
# print("rank(X_hat):", info["rankX"])
#
#
# X_pinv = np.linalg.pinv(X)
#
# BA_prime = np.zeros((4, 8, 8), dtype=np.complex128)
# for s in range(4):
#     print("X:", X.shape)
#     print("X_pinv:", X_pinv.shape)
#     print("BA[s]:", BA[s].shape)
#     BA_prime[s] = X_pinv @ BA[s] @ X
#     print("error:", s, tensors_close(BA_prime[s], DAB_AB[s]))
#     print("redidue:", norm(X @ DAB_AB[s] - BA[s] @ X, 'fro') ** 2)
#     # print(BA_prime[s])
#     # print(DAB_AB[s])
#     test_1 = X @ DAB_AB[s] @ X_pinv
#     test_2 = BA[s]
#     print(test_1)
#     print(test_1 - test_2)
#
#     print("trace:", np.trace(X_pinv @ BA[s] @ X), np.trace(DAB_AB[s]))



# eigvals, eigvecs = eigh(DAB_AB[:, :, 1])
# print(eigvals)
# print(eigvecs)
# print(np.trace(DAB_AB[:, :, 0]))
# print(np.trace(BA_double[:, :, 0]))

# # ---------- Example tensors ----------
# # A: shape (d, chi_A, chi_A)
# A = np.empty((2, 2, 2), dtype=complex)
# A[0] = np.array([[0, 1],
#                  [1, 0]], dtype=complex)          # sigma_x
# A[1] = np.array([[1, 0],
#                  [0, -1]], dtype=complex)         # sigma_z
#
# # Ground-truth rectangular intertwiner X_true: (chi_B x chi_A) with chi_B=3, chi_A=2
# X_true = np.array([[1, 0],
#                    [0, 1],
#                    [0, 0]], dtype=complex)        # tall, full column-rank
# X_pinv = np.linalg.pinv(X_true)                   # Moore-Penrose pseudo-inverse
#
# # Build B so that X_true A^s = B^s X_true
# B = np.empty((2, 3, 3), dtype=complex)
# for s in range(2):
#     B[s] = X_true @ A[s] @ X_pinv
#
# # ---------- Use your find_intertwiner(A, B) from earlier ----------
# # Paste your function definition here, then:
# X_hat, info = find_intertwiner(A, B)
#
# print("Residual (should be ~0):", info["residual"])
# print("Nullity (injective case -> 1):", info["nullity"])
# print("rank(X_hat):", info["rankX"])

# Align X_hat with X_true up to a scalar:
# lam = np.vdot(X_true.ravel(), X_hat.ravel()) / np.vdot(X_true.ravel(), X_true.ravel())
# diff = np.linalg.norm(X_hat - lam * X_true, 'fro')
# print("||X_hat - λ X_true||_F:", diff)
#
# # Verify intertwining explicitly:
# errs = [np.linalg.norm(X_hat @ A[s] - B[s] @ X_hat, 'fro') for s in range(2)]
# print("per-s Frobenius errors:", errs)
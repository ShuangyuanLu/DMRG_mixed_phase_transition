import numpy as np
import scipy
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh, eig
from partial_trace import partial_trace
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functools import partial
import math

class OpenCircuits:
    def __init__(self):
        self.id = np.eye(2)
        self.sx = np.array([[0, 1], [1, 0]])
        self.sy = np.array([[0, -1j], [1j, 0]])
        self.sz = np.array([[1, 0], [0, -1]])

        self.dim = 4
        self.rho_env = np.kron(self.id + self.sz, self.id + self.sz) / 4
        self.Hamiltonian = -(np.kron(self.sx, self.sx) + np.kron(self.sy, self.sy) + 2 * np.kron(self.sz, self.sz))
        self.Hamiltonian = np.kron(self.Hamiltonian, np.kron(self.id, self.id)) + np.kron(np.kron(self.id, self.id), self.Hamiltonian)

        self.state = np.diag([0.6, 0.25, 0.1, 0.05])

    def run(self):
        np.random.seed(0)
        # initial_parameters = np.zeros(self.dim ** 4 + self.dim ** 4) + 0.1 * np.random.random(self.dim ** 4 + self.dim ** 4)
        initial_parameters = np.zeros(self.dim ** 4) + 0.1 * np.random.random(self.dim ** 4)
        # print("initial_difference:", self.difference_evolution_hamiltonian(initial_parameters))
        print("initial_difference:", self.difference_ness_state(initial_parameters))

        # print(self.func(initial_parameters))
        # result = minimize(self.difference_evolution_hamiltonian, initial_parameters, method='BFGS', options={'disp': True})
        result = minimize(self.difference_ness_state, initial_parameters, method='BFGS', options={'disp': True})
        print("Optimal parameters:", result.x)
        print("Minimum value of the function:", result.fun)

        unitary_matrix = OpenCircuits.generate_unitary_matrix(result.x, self.dim ** 2)
        evolution_matrix = self.evolution_matrix(unitary_matrix, self.rho_env)
        eig_ness, ness = eigs(evolution_matrix, k=1, which='LM')
        ness = ness.reshape((self.dim, self.dim))
        ness = ness / np.trace(ness)
        print(eig_ness)
        print(ness)

        '''
        unitary_matrix = OpenCircuits.generate_unitary_matrix(result.x[0: self.dim ** 4], self.dim ** 2)
        general_time_reversal_matrix = scipy.linalg.expm(OpenCircuits.generate_general_time_reversal_matrix(result.x[self.dim ** 4:], self.dim ** 2))
        optimized_matrix = self.evolution_matrix(unitary_matrix, self.rho_env)
        eigenvalues, eigenstates = eig(optimized_matrix)
        np.set_printoptions(linewidth=200)
        '''

        #print(optimized_matrix)
        #print(general_time_reversal_matrix)

        #print(eigenvalues)
        #print(np.round(eigenstates, 5))
        #print(eigenstates)

    def evolution(self, rho_sys_input, U, rho_env):
        rho_sys_input = np.reshape(rho_sys_input, [self.dim, self.dim])
        rho_sys_output = partial_trace(np.matmul(np.matmul(U, np.kron(rho_env, rho_sys_input)), (U.conj().transpose())),
                                       1, [self.dim, self.dim])
        rho_sys_output = np.reshape(rho_sys_output, [self.dim ** 2])
        return rho_sys_output

    def evolution_matrix(self, U, rho_env):
        evolution_function = partial(self.evolution, U=U, rho_env=rho_env)
        evolution_operator = LinearOperator((self.dim ** 2, self.dim ** 2), matvec=evolution_function)
        return OpenCircuits.linear_operator_to_matrix(evolution_operator, self.dim ** 2)

    def difference_evolution_hamiltonian(self, input_parameters):       # parameters size 2 * self.dim ** 4
        #print("input_parameters:", np.size(input_parameters))
        parameters_unitary_matrix = input_parameters[0: self.dim ** 4]
        #print("parameters_unitary_matrix:", np.size(parameters_unitary_matrix))
        unitary_matrix = OpenCircuits.generate_unitary_matrix(parameters_unitary_matrix, self.dim ** 2)

        parameters_general_time_reversal_matrix = input_parameters[self.dim ** 4:]
        #print("parameters_general_time_reversal_matrix:", np.size(parameters_general_time_reversal_matrix))
        general_time_reversal_matrix = scipy.linalg.expm(OpenCircuits.generate_general_time_reversal_matrix(parameters_general_time_reversal_matrix, self.dim ** 2))
        general_time_reversal_matrix_inverse = scipy.linalg.expm(-OpenCircuits.generate_general_time_reversal_matrix(parameters_general_time_reversal_matrix, self.dim ** 2))

        evolution_matrix = self.evolution_matrix(unitary_matrix, self.rho_env)
        general_hamiltonian = np.matmul(general_time_reversal_matrix, np.matmul(self.Hamiltonian, general_time_reversal_matrix_inverse))

        difference_matrix = evolution_matrix - general_hamiltonian
        #print("difference_matrix:", np.size(difference_matrix))
        difference = np.sqrt(np.sum(difference_matrix * np.conj(difference_matrix)))
        #print("difference:", difference)
        return difference.real

    def difference_ness_state(self, input_parameters):      # parameters size self.dim ** 4
        unitary_matrix = OpenCircuits.generate_unitary_matrix(input_parameters, self.dim ** 2)
        evolution_matrix = self.evolution_matrix(unitary_matrix, self.rho_env)
        _, ness = eigs(evolution_matrix, k=1, which='LM')
        ness = ness.reshape((self.dim, self.dim))
        ness = ness / np.trace(ness)

        difference_matrix = ness - self.state

        difference = np.sqrt(np.sum(difference_matrix * np.conj(difference_matrix)))
        return difference.real

    @staticmethod
    def linear_operator_to_matrix(operator, n):
        basis_vectors = np.eye(n)  # Standard basis vectors (identity matrix)
        return np.column_stack([operator(basis_vectors[:, i]) for i in range(n)])

    @staticmethod
    def hermitian_residue(operator):
        residue = operator - np.conj(operator.T)
        return np.sqrt(np.sum(residue * np.conj(residue)))

    @staticmethod
    def generate_unitary_matrix(parameters, dim):
        matrix = parameters.reshape((dim, dim))
        matrix_dagger = np.conj(matrix.T)
        hermitian_matrix = (matrix + matrix_dagger) / 2 + 1j * (matrix - matrix_dagger) / 2
        # hermitian_matrix = np.triu(matrix) - np.tril(matrix) * 1j + np.diag(np.diag(matrix)) * (1j - 1)
        # hermitian_matrix = hermitian_matrix + np.conj(hermitian_matrix.T) + np.diag(np.diag(matrix))
        unitary_matrix = scipy.linalg.expm(-1j * hermitian_matrix)
        return unitary_matrix

    @staticmethod
    def generate_general_matrix(parameters, dim):   # size of parameters: 2 * dim ** 2
        matrix = parameters[0: dim ** 2].reshape((dim, dim)) + 1j * parameters[dim ** 2:].reshape((dim, dim))
        matrix = scipy.linalg.expm(matrix)
        return matrix

    @staticmethod
    def generate_general_time_reversal_matrix(parameters, dim):
        sqrt_dim = math.isqrt(dim)
        if dim == sqrt_dim ** 2:
            matrix = parameters.reshape((sqrt_dim, sqrt_dim, sqrt_dim, sqrt_dim))
            matrix_T = np.conj(matrix.transpose(1, 0, 3, 2))
            matrix = (matrix + matrix_T) / 2 + 1j * (matrix - matrix_T) / 2
            matrix = matrix.reshape((dim, dim))
            #matrix = scipy.linalg.expm(matrix)
        else:
            raise Exception("Dimension must integer!")
        return matrix

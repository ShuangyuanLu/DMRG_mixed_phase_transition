import numpy as np
import random

class CliffordCircuits:
    def __init__(self, hamiltonian_parameters):
        self.hamiltonian_parameters = hamiltonian_parameters
        self.N = self.hamiltonian_parameters["N"]
        self.n_gate = self.hamiltonian_parameters["n_gate"]
        self.p = self.hamiltonian_parameters["p"]
        self.basis = np.eye(2 * self.N + 1, 2 * self.N + 1, dtype=bool)
        self.g_matrix = np.array([[[[0, 0], [0, 0]], [[0, 0], [1, -1]]], [[[0, -1], [0, 1]], [[0, 1], [-1, 0]]]])
        self.measurement = []
        self.entropy_list = []
        self.subsystem = np.arange(self.N // 2)

    def run(self):
        for i_gate in range(self.n_gate):
            gate = self.generate_random_gate()
            self.update(gate)
            self.entropy_list.append(self.entropy(self.subsystem))

    def update(self, gate):
        #gate_type: 0: hadamard, 1: phase, 2: cnot, 3: measurement
        #gate = [0, 1, 2]
        match gate[0]:
            case 0:
                self.basis[:, 2 * self.N] = np.logical_xor(self.basis[:, 2 * self.N], np.logical_and(self.basis[:, gate[1]], self.basis[:, self.N + gate[1]]))
                self.basis[:, [gate[1], self.N + gate[1]]] = self.basis[:, [self.N + gate[1], gate[1]]]
            case 1:
                self.basis[:, 2 * self.N] = np.logical_xor(self.basis[:, 2 * self.N], np.logical_and(self.basis[:, gate[1]], self.basis[:, self.N + gate[1]]))
                self.basis[:, self.N + gate[1]] = np.logical_xor(self.basis[:, self.N + gate[1]], self.basis[:, gate[1]])
            case 2:
                self.basis[:, 2 * self.N] = np.logical_xor(self.basis[:, 2 * self.N], np.logical_and(np.logical_and(self.basis[:, gate[1]], self.basis[:, self.N + gate[2]]),
                                                np.logical_xor(self.basis[:, gate[2]], np.logical_xor(self.basis[:, self.N + gate[1]], True))))
                self.basis[:, gate[2]] = np.logical_xor(self.basis[:, gate[2]], self.basis[:, gate[1]])
                self.basis[:, self.N + gate[1]] = np.logical_xor(self.basis[:, self.N + gate[1]], self.basis[:, self.N + gate[2]])
            case 3:
                x_a = self.basis[self.N: 2 * self.N, gate[1]]
                if x_a.any():
                    p = np.argmax(x_a) + self.N
                    for i in range(2 * self.N):
                        if i != p and self.basis[i, gate[1]]:
                            self.row_sum(i, p)
                    self.basis[p - self.N, :] = self.basis[p, :]
                    self.basis[p, :] = False
                    self.basis[p, gate[1] + self.N] = True
                    if np.random.random() < 0.5:
                        self.basis[p, 2 * self.N] = False
                    else:
                        self.basis[p, 2 * self.N] = True
                    measurement_result = self.basis[p, 2 * self.N]
                else:
                    np.random.random()
                    self.basis[2 * self.N, :] = False
                    for i in range(self.N):
                        if self.basis[i, gate[1]]:
                            self.row_sum(2 * self.N, i + self.N)
                    measurement_result = self.basis[2 * self.N, 2 * self.N]
                self.measurement.append(1 - int(measurement_result) * 2)

    def row_sum(self, h, i):
        r = 2 * int(self.basis[h, 2 * self.N]) + 2 * int(self.basis[i, 2 * self.N])
        for j in range(self.N):
            r = r + self.g_matrix[int(self.basis[i, j]), int(self.basis[i, self.N + j]), int(self.basis[h, j]), int(self.basis[h, self.N + j])]
        if r % 4 == 0:
            self.basis[h, 2 * self.N] = False
        else:
            self.basis[h, 2 * self.N] = True
        self.basis[h, 0: 2 * self.N] = np.logical_xor(self.basis[i, 0: 2 * self.N], self.basis[h, 0: 2 * self.N])

    def generate_random_gate(self):
        if np.random.random() < self.p:
            gate_type = 3
        else:
            gate_type = np.random.randint(0, 3)
        if gate_type == 2:
            site = np.random.randint(0, self.N - 1)
            gate = [gate_type, site, site + 1]
        else:
            site = np.random.randint(0, self.N)
            gate = [gate_type, site]
        return gate

    def entropy(self, subsystem):
        #subsystem is 1 dimensional np.ndarray
        sub_basis = np.copy(self.basis[self.N: 2 * self.N, np.concatenate((subsystem, subsystem + self.N))])
        n_column = subsystem.size * 2
        row = 0
        column = 0
        while column < n_column:
            if not sub_basis[row, column]:
                if np.any(sub_basis[row: self.N, column]):
                    true_index = np.argmax(sub_basis[row: self.N, column])
                    sub_basis[[row, row + true_index]] = sub_basis[[row + true_index, row]]
                else:
                    column += 1
                    continue
            true_indices = np.where(sub_basis[row + 1: self.N, column])[0]
            sub_basis[row + 1 + true_indices, column: n_column] ^= sub_basis[row, column: n_column]
            row += 1
            column += 1
        return (row - subsystem.size) * np.log(2)

    def print_basis(self):
        basis_string = np.full((2 * self.N, self.N + 1), '', dtype='U10')
        for i in range(2 * self.N):
            for j in range(self.N):
                match (bool(self.basis[i, j]), bool(self.basis[i, self.N + j])):
                    case (False, False):
                        string_ij = ''
                    case (True, False):
                        string_ij = 'X'
                    case (False, True):
                        string_ij = 'Z'
                    case (True, True):
                        string_ij = 'Y'
                    case _:
                        string_ij = 'error'
                basis_string[i, j] = string_ij
        for i in range(2 * self.N):
            if self.basis[i, 2 * self.N]:
                phase_string_i = '-1'
            else:
                phase_string_i = '1'
            basis_string[i, self.N] = phase_string_i
        print(basis_string)

        



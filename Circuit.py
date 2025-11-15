import numpy as np
from scipy import sparse

class Circuit:
    def __init__(self, hamiltonian_parameters):
        self.hamiltonian_parameters = hamiltonian_parameters
        self.N = self.hamiltonian_parameters["N"]
        self.n_gate = self.hamiltonian_parameters["n_gate"]
        self.p = self.hamiltonian_parameters["p"]
        self.state = np.zeros(2 ** self.N)
        self.state[0] = 1
        self.measurement = []
        self.entropy_list = []
        self.subsystem = np.arange(self.N // 2)

    def run(self):
        for i in range(self.n_gate):
            gate, gate_matrix = self.generate_random_clifford_gate()
            self.update(gate, gate_matrix)
            self.entropy_list.append(self.entropy(self.subsystem))

    def update(self, gate, gate_matrix):
        # 0: single-bit gate, 2: two-bit gate, 3: single-bit measurement
        match gate[0]:
            case 0:
                operator = sparse.kron(sparse.identity(2 ** gate[1]), sparse.kron(gate_matrix, sparse.identity(2 ** (self.N - gate[1] - 1))))
                self.state = operator.dot(self.state)
            case 2:
                operator = sparse.kron(sparse.identity(2 ** gate[1]), sparse.kron(gate_matrix, sparse.identity(2 ** (self.N - gate[1] - 2))))
                self.state = operator.dot(self.state)
            case 3:
                operator = sparse.kron(sparse.identity(2 ** gate[1]), sparse.kron(gate_matrix, sparse.identity(2 ** (self.N - gate[1] - 1))))
                expectation = np.dot(self.state.conj(), operator.dot(self.state))
                if np.random.random() < (expectation + 1) / 2:
                    projector = (sparse.identity(2 ** self.N) + operator) / 2
                    measurement_result = 1
                else:
                    projector = (sparse.identity(2 ** self.N) - operator) / 2
                    measurement_result = -1
                self.state = projector.dot(self.state)
                self.state = self.state / np.sqrt(np.dot(self.state.conj(), self.state))
                self.measurement.append(measurement_result)
            case _:
                raise ValueError("No such gate.")

    def generate_random_clifford_gate(self):
        if np.random.random() < self.p:
            gate_type = 3
        else:
            gate_type = np.random.randint(0, 3)
        if gate_type == 2:
            site = np.random.randint(0, self.N - 1)
            gate = [gate_type, site, site + 1]
            gate_matrix = sparse.csr_array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        else:
            site = np.random.randint(0, self.N)
            gate = [gate_type, site]
            if gate_type == 0:
                gate_matrix = sparse.csr_array([[1, 1], [1, -1]]) / np.sqrt(2)
            elif gate_type == 1:
                gate_matrix = sparse.csr_array([[1, 0], [0, 1j]])
                gate[0] = 0
            elif gate_type == 3:
                gate_matrix = sparse.csr_array([[1, 0], [0, -1]])
            else:
                raise ValueError("No such gate.")
        return gate, gate_matrix

    def entropy(self, subsystem):
        n_subsystem = subsystem.size
        state = self.state.reshape(2 ** n_subsystem, 2 ** (self.N - n_subsystem))
        density_matrix = np.matmul(state.conj(), state.transpose())
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        epsilon = 10 ** (-15)
        filtered_eigenvalues = eigenvalues[eigenvalues > epsilon]
        entropy = -sum(filtered_eigenvalues * np.log(filtered_eigenvalues))
        return entropy

    def double_check_clifford_circuit(self, clifford_circuit):
        expectation_list = np.zeros(2 * clifford_circuit.N)
        for i in range(2 * clifford_circuit.N):
            operator = sparse.identity(1)
            for j in range(clifford_circuit.N):
                match (bool(clifford_circuit.basis[i, j]), bool(clifford_circuit.basis[i, j + clifford_circuit.N])):
                    case (False, False):
                        operator_j = sparse.identity(2)
                    case (False, True):
                        operator_j = sparse.csr_matrix([[1, 0], [0, -1]])
                    case (True, False):
                        operator_j = sparse.csr_matrix([[0, 1], [1, 0]])
                    case (True, True):
                        operator_j = sparse.csr_matrix([[0, -1j], [1j, 0]])
                    case _:
                        raise ValueError("Mistake.")
                operator = sparse.kron(operator, operator_j)
            if clifford_circuit.basis[i, 2 * clifford_circuit.N]:
                phase = -1
            else:
                phase = 1
            expectation_list[i] = np.dot(self.state.conj(), operator.dot(self.state)).real * phase
        print(expectation_list)


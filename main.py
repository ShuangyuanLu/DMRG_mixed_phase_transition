import numpy as np
import matplotlib.pyplot as plt
from Circuit import Circuit
from CliffordCircuits import CliffordCircuits
from ZXZ_chain import ZXZ_chain
from gate_map import gate_map
from OpenCircuits import OpenCircuits
from DMRG import DMRG
from DMRG_Ising import DMRG_Ising
from DMRG_NESS_Ising import DMRG_NESS_Ising
from tDMRG_NESS_Ising import tDMRG_NESS_Ising
import csv
from tDMRG_NESS_Heisenberg import tDMRG_NESS_Heisenberg
import time
import os
import shelve
from concurrent.futures import ProcessPoolExecutor, as_completed
from Spin_1_Heisenberg_chain import Spin_1_Heisenberg_chain
from zxz_gapless_chain import ZXZ_gapless_chain


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)

    parameters = {'J': 1, 'h': 0.2, 'K': 0.5, 'L': 20, "cutoff_s": 10 ** (-8), "cutoff_n": 40, "n_sweep": 12}
    my_model = ZXZ_gapless_chain(parameters)
    start_time = time.perf_counter()
    my_model.run()
    end_time = time.perf_counter()

    print("time:", round(end_time - start_time, 2))
    print(my_model.result)

    for i in range(parameters['L']):
        print(my_model.state[i].shape)



    '''
    parameters = {'h': 1, 'J': 0.3, 'J1': 0.5, 'L': 8, "cutoff_s": 10 ** (-8), "cutoff_n": 50, "n_sweep": 100}
    my_dmrg = tDMRG_NESS_Heisenberg(parameters)
    my_dmrg.run()

    id = np.eye(2)
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    #measurement_result = my_dmrg.measure_site(parameters['L'] // 2, sx)
    #measurement_result = my_dmrg.measure_2_sites(parameters['L'] // 2, parameters['L'] // 2 + 1, sz)
    measurement_result = my_dmrg.measure_time_corr(parameters['L'] // 2, parameters['L'] // 2, np.kron(sz, sz))
    print("expectation value:", measurement_result)
    '''


    #gate_map()
    #my_open_circuit = OpenCircuits()
    #my_open_circuit.run()


    #ising_dmrg = DMRG_Ising()
    #ising_dmrg.run()
    #print(ising_dmrg.result)
    #print(ising_dmrg.measure_site(5, np.array([[1, 0], [0, -1]])))
    #print(ising_dmrg.measure_site(5, np.array([[0, 1], [1, 0]])))
    '''
    order_parameter_list = []
    h_list = [0.3 + 0.025 * i for i in range(1)]

    for i_h in range(len(h_list)):
        parameters = {'h': h_list[i_h], 'J': 1, 'L': 8, "cutoff_s": 10 ** (-8), "cutoff_n": 50, "n_sweep": 500}
        dmrg_ness_ising = tDMRG_NESS_Ising(parameters)
        #for i in range(dmrg_ness_ising.L):
        #    print(dmrg_ness_ising.hamiltonian[i].shape)
        dmrg_ness_ising.run()

        #print(dmrg_ness_ising.result)
        print('h:', h_list[i_h])
        for i in range(dmrg_ness_ising.L):
            print(dmrg_ness_ising.state[i].shape)
        #print(dmrg_ness_ising.trace())
        #print(np.round(dmrg_ness_ising.state[3], 4))
        #print("evolution:", np.round(dmrg_ness_ising.evolution[1][0], 3))
        #print(np.sum(np.abs(dmrg_ness_ising.evolution[1][0]) > 0.5))

        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])
        #print('measure_sx:', dmrg_ness_ising.measure_site(2, sx))

        sx_expectation = dmrg_ness_ising.measure_site(dmrg_ness_ising.L // 2, sx)
        sz_expectation = dmrg_ness_ising.measure_site(0, sz)
        sz_expectation_1 = dmrg_ness_ising.measure_site(dmrg_ness_ising.L - 1, sz)
        sz_corr = dmrg_ness_ising.measure_2_sites(0, dmrg_ness_ising.L - 3, sz)
        print("sx:", sx_expectation)
        print("sz:", sz_expectation)
        print("sz:", sz_expectation_1)
        print("sz_corr:", sz_corr)

    #    order_parameter_list.append(np.real(sx_expectation))
    #    print(sx_expectation)

    #with open("data.csv", mode="w", newline="") as file:
    #    writer = csv.writer(file)
    #    writer.writerows([[value] for value in order_parameter_list])
    '''

'''
    hamiltonian_parameters = {"N": 100, "n_gate": 100000, "p": 0.03}
    np.random.seed(0)
    my_clifford_circuit = CliffordCircuits(hamiltonian_parameters)
    my_clifford_circuit.run()
    #my_clifford_circuit.print_basis()
    #print(my_clifford_circuit.entropy_list)
    plt.plot(my_clifford_circuit.entropy_list)
    plt.show()



    #np.random.seed(0)
    #my_circuit = Circuit(hamiltonian_parameters)
    #my_circuit.run()

    #my_circuit.double_check_clifford_circuit(my_clifford_circuit)

    #print(my_circuit.measurement)
    #print(my_clifford_circuit.measurement)
'''



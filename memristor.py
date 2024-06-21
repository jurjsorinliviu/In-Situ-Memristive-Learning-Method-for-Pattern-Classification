# memristor from https://arxiv.org/pdf/1711.06819

import matplotlib.pyplot as plt
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
import os
import numpy as np

def get_all_nodes_Cmem(circuit):
    # this only works for the first level of subcircuits
    nodes = []
    for subcircuit in circuit.subcircuits:
        for node in subcircuit.nodes:
            if node.name.lower() == 'cmem':
                nodes.append(("X"+subcircuit.name+"."+node.name).lower())
    return nodes

def get_all_nodes_recursive(circuit, currentName):
    nodes = []
    for subcircuit in circuit.subcircuits:
        if currentName == "":
            subcircuitName = "X" + subcircuit.name
        else:
            subcircuitName = currentName + ".X" + subcircuit.name
        for node in subcircuit.nodes:
            if node.name != circuit.gnd:
                nodes.append(subcircuitName + "." + node.name)
        nodes += get_all_nodes_recursive(subcircuit, subcircuitName)
    return nodes

def get_all_cmem_nodes_recursive(circuit):
    all_nodes = get_all_nodes_recursive(circuit, "")
    cmem_nodes = []
    for node in all_nodes:
        if node.lower().find("cmem") != -1:
            cmem_nodes.append(node)
    return cmem_nodes

def add_initial_condition(simulator, circuit):
    kwargDict = {}
    IC = 0
    for n in get_all_cmem_nodes_recursive(circuit):
        kwargDict[n] = IC
    print(kwargDict)
    simulator.initial_condition(**kwargDict)


class Memristor(SubCircuit):
    def __init__(self, name, K=0.00005, Kp=0.13*10**-3, C=0.1@u_pF, Vto=0, SwitchRoff=10**20, SwitchRon=10**-3):
        __nodes__ = ('S', 'D', 'phi')
        SubCircuit.__init__(self, name, *__nodes__)
        # self.parameters = {'K': K, 'Kp': Kp, 'C': C, 'Vto': Vto}
        self.VCCS('m', self.gnd, 'o', 'D', 'S', K)
        self.R('R2', 'o', self.gnd, 10**12)
        self.VoltageControlledSwitch('sw', 'o', 'Cmem', 'phi', self.gnd, model='my_switch')
        self.model('my_switch', "SW", Ron=SwitchRon, Roff=SwitchRoff, Vt=0.5)
        self.C('C', self.gnd, 'Cmem', C)
        self.model('my_nmos', 'NMOS', level=3, Vto=Vto, Kp=Kp)
        # M1 drain gate source body
        self.MOSFET(1, 'D', 'Cmem', 'S', self.gnd, model='my_nmos')

class MemristorsInSeries(SubCircuit):
    def __init__(self, name, K=0.00045, Kp=0.13*10**-3, C=0.1@u_pF, Vto=0, N=3):
        __nodes__ = ('S', 'D')
        SubCircuit.__init__(self, name, *__nodes__)
        # self.parameters = {'K': K, 'Kp': Kp, 'C': C, 'Vto': Vto}
        self.subcircuit(Memristor('Mem0', K=K, Kp=Kp, C=C, Vto=Vto))
        self.X('Mem0', 'mem0', 'S', 'o0')
        for i in range(1, N):
            self.subcircuit(Memristor('Mem'+str(i), K=K, Kp=Kp, C=C, Vto=Vto))
            self.X('Mem'+str(i), 'mem'+str(i), 'o'+str(i-1), 'o'+str(i))
        self.subcircuit(Memristor('Mem'+str(N), K=K, Kp=Kp, C=C, Vto=Vto))
        self.X('Mem'+str(N), 'mem'+str(N), 'o'+str(N-1), 'D')

def testMemristorInSeries():
    # memristors in series test
    vs = 0
    vd = 0.2
    circuit = Circuit('NMOS Transistor')
    circuit.V('S', 'S', circuit.gnd, vs@u_V)
    circuit.SinusoidalVoltageSource('D', 'D', circuit.gnd, amplitude=vd@u_V, frequency=1@u_kHz)
    circuit.subcircuit(MemristorsInSeries('SMem1', K=0.00045, Kp=0.13*10**-3, C=0.1@u_pF, Vto=0, N=3))
    circuit.X('SMem1', 'smem1', 'S', 'D')

    print(circuit)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    add_initial_condition(simulator, circuit)
    # kwargDict = {}
    # IC = 0
    # for n in get_all_cmem_nodes_recursive(circuit):
    #     kwargDict[n] = IC
    # print(kwargDict)
    # simulator.initial_condition(**kwargDict)
    analysis = simulator.transient(step_time=0.0001@u_ms, end_time=10@u_ms)

    # plot drain current at vd
    plt.plot(np.array(analysis.branches['vd']), label='Id')
    plt.legend()
    plt.show()

    # vd vs ID
    vds = np.array(analysis['D']) - np.array(analysis['S'])

    figure, ax = plt.subplots(figsize=(20, 10))
    plt.plot(vds, -np.array(analysis.branches['vd']), label='ID')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    vs = 0
    vd = 1
    phi = 1.2
    circuit = Circuit('NMOS Transistor')
    circuit.V('S', 'S', circuit.gnd, vs@u_V)
    circuit.SinusoidalVoltageSource('D', 'D', circuit.gnd, amplitude=vd@u_V, frequency=1@u_kHz, offset=0@u_V)
    circuit.subcircuit(Memristor('Mem1', K=0.0002, Kp=0.13*10**-3, C=0.1@u_pF, Vto=0))
    circuit.V('phi', 'phi', circuit.gnd, phi@u_V)
    circuit.X('Mem1', 'mem1', 'S', 'D', 'phi')

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    # IMPORTANT! the initial condition is neccesary for capacitor to work
    add_initial_condition(simulator, circuit)

    analysis = simulator.transient(step_time=0.0001@u_ms, end_time=10@u_ms, use_initial_condition=True)

    # vd vs ID
    vds = np.array(analysis['D']) - np.array(analysis['S'])
    plt.plot(vds, np.array(analysis.branches['vs']))
    plt.legend()
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title("SPICE simulation of memristor IV curve")
    plt.savefig('memristor_circuit_simulation.png')
    plt.show()
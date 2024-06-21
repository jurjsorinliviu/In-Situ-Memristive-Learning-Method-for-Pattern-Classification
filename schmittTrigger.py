import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from PySpice.Unit import *
from PySpice.Probe.Plot import plot

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


class SchmittTrigger(SubCircuit):
    def __init__(self, name, maxVoltage=1@u_V, threshold=0.5@u_V, vcc_offset=0, vcc_n_offset=0):
        __nodes__ = ('vin', 'vout2')
        SubCircuit.__init__(self, name, *__nodes__)
        self.R(1, 1, self.gnd, 100@u_Ω)
        self.R(2, 'vout', 1, 100@u_Ω)
        self.V('+vcc', '+Vcc', self.gnd, maxVoltage+vcc_offset@u_V)
        self.V('-vcc', '-Vcc', self.gnd, vcc_n_offset@u_V)
        # circuit.V('vb', 'vb', circuit.gnd, 0@u_V)
        self.X(f'op_amp', 'uopamp_lvl2', 'vin', self.gnd, f'+Vcc', '-Vcc', 'vout', Ilimit=100@u_MA, Vos=-threshold, Vmax=1@u_MV, Iq=1@u_MA, GBW=10**10@u_Hz, Rout=1@u_Ω, Avol=10**5)
        
        self.V('+vcc2', '+Vcc2', self.gnd, maxVoltage+vcc_offset@u_V)
        self.V('-vcc2', '-Vcc2', self.gnd, vcc_n_offset@u_V)
        self.X(f'op_amp2', 'uopamp_lvl2', 'vout', self.gnd, f'+Vcc2', '-Vcc2', 'vout2', Ilimit=100@u_MA, Vos=-maxVoltage/2, Vmax=1@u_MV, Iq=1@u_MA, GBW=10**10@u_Hz, Rout=1@u_Ω, Avol=10**5)
        # V = +- 0.63*Vcc
        # self.R('load', 2, self.gnd, 100@u_Ω)

if __name__ == "__main__":
    circuit = Circuit('Layer Connections')
    # libraries_path = find_libraries()
    # spice_library = SpiceLibrary(libraries_path)
    # circuit.include(spice_library['2n2222a'])
    # circuit.include('2n2222a.lib')
    circuit.include("uopamp_v1.1.lib")

    # circuit.V(1, 1, circuit.gnd, 1@u_V)

    # circuit.R("schmitt", 'sch_1', 1, 200@u_kΩ)
    # circuit.R("schmitt2", 2, 'sch_1', 10@u_kΩ)
    # circuit.V('+vcc', '+Vcc', circuit.gnd, 1@u_V)
    # # -vcc voltage
    # circuit.V('-vcc', '-Vcc', circuit.gnd, -1@u_V)
    # circuit.X(f'op_amp', 'uopamp_lvl2', "sch_1", circuit.gnd, f'+Vcc', '-Vcc', 2)
    # # load
    # circuit.R('load', 2, circuit.gnd, 1@u_kΩ)
    voltages = []
    vals = np.arange(-1, 2, 0.01)
    for val in vals:
        circuit = Circuit('Layer Connections')
        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)
        # circuit.include(spice_library['2n2222a'])
        # circuit.include('2n2222a.lib')
        circuit.include("uopamp_v1.1.lib")
        circuit.subcircuit(SchmittTrigger('sch1', maxVoltage=1))
        circuit.V(1, 'oin', circuit.gnd, val@u_V)
        circuit.X(1, 'sch1', 'oin', 2)
        # circuit.R(1, 1, circuit.gnd, 100@u_Ω)
        # circuit.R(2, 2, 1, 100@u_Ω)
        # circuit.V('+vcc', '+Vcc', circuit.gnd, 2@u_V)
        # circuit.V('-vcc', '-Vcc', circuit.gnd, -0.0717256@u_V)
        # # circuit.V('vb', 'vb', circuit.gnd, 0@u_V)
        # circuit.X(f'op_amp', 'uopamp_lvl2', 'oin', circuit.gnd, f'+Vcc', '-Vcc', 2, Ilimit=100@u_MA, Vos=-0.5@u_V, Vmax=1@u_MV, Iq=1@u_MA, GBW=10**10@u_Hz, Rout=1@u_Ω, Avol=10**5)
        # # V = +- 0.63*Vcc
        circuit.R('load', 2, circuit.gnd, 100@u_Ω)

        # fer implementation
        # circuit.V(1, 'oin', circuit.gnd, val@u_V)
        # circuit.V('+Vcc', '+Vcc', circuit.gnd, 10@u_V)
        # circuit.V('-Vcc', '-Vcc', circuit.gnd, -10@u_V)
        # circuit.X(f'op_amp', 'uopamp_lvl2', 2, 'oin', '+Vcc', '-Vcc', 1, Ilimit=100@u_MA, Vos=0@u_V, Vmax=1@u_MV, Iq=1@u_A, GBW=10**10@u_Hz, Rout=1@u_Ω, Avol=10**5)
        # circuit.R(1, 1, 2, 100@u_Ω)
        # circuit.R(2, 2, circuit.gnd, 100@u_Ω)
        # circuit.R('load', 1, circuit.gnd, 100@u_Ω)
        print(circuit)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        startValue = -1
        endValue = 10
        stepValue = 0.1
        # analysis = simulator.dc(V1=slice(startValue, endValue, stepValue))
        # tran analysis
        analysis = simulator.transient(step_time=1@u_ms, end_time=5@u_ms)
        # voltages.append(np.array(analysis.nodes['2'])[-1])

        # plot voltage at node 2
        # map res to a color between red and blue
        # color = (res - 100) / (10000 - 1)
        # plt.plot(np.arange(startValue, endValue+stepValue, stepValue), np.array(analysis.nodes['2']))
        voltages.append(np.array(analysis.nodes['2'][-1]))
    plt.plot(vals, voltages)
    # draw a dotted line at y = 0
    plt.axhline(y=0, color='k', linestyle='--')
    # draw a dotted line at y = 1
    plt.axhline(y=1, color='k', linestyle='--')
    plt.show()
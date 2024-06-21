import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from PySpice.Unit import *
from PySpice.Probe.Plot import plot

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

from schmittTrigger import SchmittTrigger


class MemristorCrossbar(SubCircuit):
    def __init__(self, name, num_inputs=None, num_outputs=None, resistancesP=None, resistancesN=None, biasP=None, biasN=None):
        # assert name is string, num_inputs and num_outputs are integers, resistancesP and resistancesN are numpy arrays of size num_inputs x num_outputs and biasR is a numpy array of size num_outputs
        assert isinstance(name, str), 'name must be a string'
        assert isinstance(num_inputs, int), 'num_inputs must be an integer'
        assert isinstance(num_outputs, int), 'num_outputs must be an integer'
        assert isinstance(resistancesP, np.ndarray), 'resistancesP must be a numpy array'
        assert isinstance(resistancesN, np.ndarray), 'resistancesN must be a numpy array'
        assert isinstance(biasP, np.ndarray), 'biasR must be a numpy array'
        assert isinstance(biasN, np.ndarray), 'biasR must be a numpy array'
        assert resistancesP.shape == (num_inputs, num_outputs), f'resistancesP must be a numpy array of size {num_inputs} x {num_outputs} (num_inputs x num_outputs), not {resistancesP.shape}'
        assert resistancesN.shape == (num_inputs, num_outputs), f'resistancesN must be a numpy array of size {num_inputs} x {num_outputs} (num_inputs x num_outputs), not {resistancesN.shape}'
        assert biasP.shape == (num_outputs,), f'biasR must be a numpy array of size {num_outputs} (num_outputs), not {biasP.shape}'
        assert biasN.shape == (num_outputs,), f'biasR must be a numpy array of size {num_outputs} (num_outputs), not {biasN.shape}'
        # assert all values are positive
        assert np.all(resistancesP >= 0), 'resistancesP must be positive'
        assert np.all(resistancesN >= 0), 'resistancesN must be positive'

        # print(name)
        # print("Weights:")
        # print(resistancesP)
        # print()
        # print(resistancesN)
        # print()
        # print("Biases:")
        # print(biasP)
        # print()
        # print(biasN)
        inputNodesNames = [f'input_{i}' for i in range(num_inputs)]
        outputNodesNames = [f'y_{j}' for j in range(num_outputs)]
        __nodes__ = inputNodesNames + outputNodesNames
        SubCircuit.__init__(self, name, *__nodes__)

        self.V(f'bias', f'bias', self.gnd, 1@u_V)
        # Create output nodes
        output_nodes = []
        for j in range(num_outputs):
            output_node = self.get_node(f'output_p_{j}', True)
            output_nodes.append(output_node)
            output_node = self.get_node(f'output_n_{j}', True)
            output_nodes.append(output_node)

        # Create memristor crossbar connections for positive weights
        for i in range(num_inputs):
            for j in range(num_outputs):
                # Use resistors to model memristors
                resistance = resistancesP[i, j]@u_kΩ
                res = self.R(f'Rp_{i}_{j}', f'input_{i}', f'output_p_{j}', resistance)
                resistance = resistancesN[i, j]@u_kΩ
                res = self.R(f'Rn_{i}_{j}', f'input_{i}', f'output_n_{j}', resistance)

        # repeat for bias resistors
        for j in range(num_outputs):
            resistance = biasP[j]@u_kΩ
            res = self.R(f'bias_p_{j}', f'bias', f'output_p_{j}', resistance)
            resistance = biasN[j]@u_kΩ
            res = self.R(f'bias_n_{j}', f'bias', f'output_n_{j}', resistance)

        # connect the output pairs to op amps
        for j in range(num_outputs):
            # use a sigmoid neuron for each output
            self.subcircuit(LapiqueNeuron(f'neuron_{j}', 10, 0.00015, 5))
            self.X(f'neuron_{j}', f'neuron_{j}', f'output_p_{j}', f'output_n_{j}', f'y_{j}')


class SigmoidNegPosNeuron(SubCircuit):
    def __init__(self, name, vcc=20@u_V, vcc2=0.5@u_V, vcc2_offset=0@u_V, res=1@u_mΩ, rf=1@u_kΩ):
        __nodes__ = ('input+', 'input-', 'output')
        SubCircuit.__init__(self, name, *__nodes__)
        self.V(f'Vcc', f'+Vcc', self.gnd, vcc)
        self.V(f'Vcc-', f'-Vcc', self.gnd, -vcc)
        self.X(f'op_amp_p', 'uopamp_lvl2', self.gnd, f'input+', f'+Vcc', '-Vcc', f'N_j')
        self.R(f'op_amp', f'N_j', f'input+', res)
        self.R(f'op_amp2', f'N_j', f'input-', res)
        self.V(f'Vcc2', f'+Vcc2', self.gnd, vcc2+vcc2_offset)
        self.V(f'Vcc2-', f'-Vcc2', self.gnd, -vcc2+vcc2_offset)
        self.X(f'op_amp_n', 'uopamp_lvl2', self.gnd, f'input-', f'+Vcc2', '-Vcc2', f'output')
        self.R(f'_f', f'output', f'input-', rf)

class LapiqueNeuron(SubCircuit):
    def __init__(self, name, R, C, threshold):
        SubCircuit.__init__(self, name, 'in_p', 'in_n', 'out')
        # add resistances of 1k to the inputs
        # self.R('1', 'in_p', 'in_p_', 1@u_kΩ)
        # self.R('2', 'in_n', 'in_n_', 1@u_kΩ)
        # substract the two inputs
        self.subcircuit(SigmoidNegPosNeuron('sigmoid', vcc=20@u_V, vcc2=20@u_V, vcc2_offset=0.5@u_V, res=1@u_kΩ, rf=1@u_kΩ))
        self.X('sigmoid', 'sigmoid', 'in_p', 'in_n', '1')
        # self.X('sigmoid', 'sigmoid', 'in_p_', 'in_n_', 'out')

        # convert to current

        # circuit.VCCS(1, 1, circuit.gnd, circuit.gnd, 'in_',  1)
        self.VCCS('VCCS', '2', self.gnd, self.gnd, '1', 1)
        # go through RC circuit
        self.R('R', '2', self.gnd, R@u_Ω)
        self.C('C', '2', self.gnd, C@u_F)
        # TODO test this last 3 components on their own to compare with real life implementation

        self.subcircuit(SchmittTrigger('sch1', maxVoltage=1@u_V, threshold=threshold@u_V, vcc_offset=0.07, vcc_n_offset=-0.07))
        self.X(1, 'sch1', '2', 'out')

def sigmoidTest():
    # test sigmoid neuron with 0 voltage in_n and dc voltage in_p
    circuit = Circuit('Sigmoid Neuron')
    circuit.include("uopamp_v1.1.lib")
    circuit.V('input', 'in_p', circuit.gnd, 1@u_V)
    circuit.R('1', 'p', 'in_p', 1@u_kΩ)
    circuit.V('input_', 'in_n', circuit.gnd, 0@u_V)
    circuit.R('2', 'n', 'in_n', 1@u_kΩ)
    circuit.subcircuit(SigmoidNegPosNeuron('neuron', 20@u_V, 20@u_V, 0@u_V, 1@u_kΩ, 1@u_kΩ))
    circuit.X('1', 'neuron', 'p', 'n', 'out')
    # load
    circuit.R('load', 'out', circuit.gnd, 470@u_Ω)

    print(circuit)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vinput=slice(0, 1, 0.01))

    figure, ax = plt.subplots(figsize=(20, 10))

    ax.plot(np.array(analysis['in_p']), np.array(analysis['out']), label='V1')

    plt.show()

def lapiqueTest():
    # test lapique neuron
    x_n = 10
    circuit = Circuit('Lapique Neuron')
    circuit.include("uopamp_v1.1.lib")
    circuit.V('input', 'in_p', circuit.gnd, 1@u_V)
    circuit.R('1', 'p', 'in_p', 1@u_kΩ)
    circuit.V('input_', 'in_n', circuit.gnd, x_n@u_V)
    circuit.R('2', 'n', 'in_n', 1@u_kΩ)
    circuit.subcircuit(LapiqueNeuron('neuron', 10, 0.00015, 5))
    circuit.X('1', 'neuron', 'p', 'n', 'out')
    # load
    circuit.R('load', 'out', circuit.gnd, 470@u_Ω)

    print(circuit)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vinput=slice(0, 15, 0.01))

    figure, ax = plt.subplots(figsize=(20, 10))

    ax.plot(np.array(analysis['in_p']), np.array(analysis['out']), label='V1')
    # show a dotted line at y = 0 and y = 1
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=1, color='k', linestyle='--')
    # and at x=x_n+0.5
    plt.axvline(x=x_n+0.5, color='r', linestyle='--')

    plt.show()

def lapiqueConnectionTest():
    # test lapique neuron connection
    x_n = 0
    circuit = Circuit('Lapique Neuron')
    circuit.include("uopamp_v1.1.lib")
    circuit.V('input', 'in_p', circuit.gnd, 1@u_V)
    # circuit.R('1', 'p', 'in_p', 1@u_kΩ)
    circuit.V('input_', 'in_n', circuit.gnd, x_n@u_V)
    # circuit.R('2', 'n', 'in_n', 1@u_kΩ)
    circuit.subcircuit(LapiqueNeuron('neuron', 10, 0.00015, 5))
    circuit.X('1', 'neuron', 'in_p', 'in_n', 'out')
    circuit.V('input2', 'in_n2', circuit.gnd, 0.45@u_V)
    # circuit.R('12', 'n2', 'in_n2', 1@u_kΩ)
    # circuit.R('22', 'out_', 'out', 1@u_kΩ)
    circuit.subcircuit(LapiqueNeuron('neuron2', 10, 0.00015, 5))
    circuit.X('2', 'neuron2', 'out', 'in_n2', 'out2')
    # load
    circuit.R('load', 'out', circuit.gnd, 1@u_kΩ)

    print(circuit)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vinput=slice(0, 3, 0.01))

    for node in analysis.nodes:
        print(node, np.array(analysis[node])[-1])

    figure, ax = plt.subplots(figsize=(20, 10))

    # ax.plot(np.array(analysis['in_p']), np.array(analysis['x1.2']), label='x1.2')
    ax.plot(np.array(analysis['in_p']), np.array(analysis['out']), label='Vinternal')
    ax.plot(np.array(analysis['in_p']), np.array(analysis['out2']), label='Vout')
    # show a dotted line at y = 0 and y = 1
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=1, color='k', linestyle='--')
    # and at x=x_n+0.5
    plt.axvline(x=x_n+0.5, color='r', linestyle='--')
    plt.legend()
    plt.show()


def generate_P_N_resistances(resistances):
    # return the positive and negative resistances, such that resistancesP-resistancesN = resistances
    num_inputs, num_outputs = resistances.shape
    conductances = 1/resistances
    conductancesP = np.zeros((num_inputs, num_outputs))
    conductancesN = np.zeros((num_inputs, num_outputs))
    for i in range(num_inputs):
        for j in range(num_outputs):
            if conductances[i, j] > 0:
                conductancesP[i, j] = conductances[i, j]
                conductancesN[i, j] = 10**-8
            elif conductances[i, j] == 0:
                conductancesP[i, j] = 10**-8
                conductancesN[i, j] = 10**-8
            else:
                conductancesN[i, j] = -conductances[i, j]
                conductancesP[i, j] = 10**-8

    # clip conductances between -100 and 100
    conductancesP = np.clip(conductancesP, -100, 100)
    conductancesN = np.clip(conductancesN, -100, 100)
    resistancesP = 1/conductancesP
    resistancesN = 1/conductancesN
    return resistancesP, resistancesN

def generate_P_N_bias(biasR):
    # return the positive and negative bias resistances, such that biasP-biasN = biasR
    num_outputs = len(biasR)
    biasConductances = 1/biasR
    biasConductancesP = np.zeros(num_outputs)
    biasConductancesN = np.zeros(num_outputs)
    for j in range(num_outputs):
        if biasConductances[j] > 0:
            biasConductancesP[j] = biasConductances[j]
            biasConductancesN[j] = 10**-10
        elif biasConductances[j] == 0:
            biasConductancesP[j] = 10**-10
            biasConductancesN[j] = 10**-10
        else:
            biasConductancesN[j] = -biasConductances[j]
            biasConductancesP[j] = 10**-10
    # clip conductances between -100 and 100
    biasConductancesP = np.clip(biasConductancesP, -100, 100)
    biasConductancesN = np.clip(biasConductancesN, -100, 100)

    biasResistancesP = 1/biasConductancesP
    biasResistancesN = 1/biasConductancesN
    return biasResistancesP, biasResistancesN

import json
# make model from json
class SNNModel(SubCircuit):
    def __init__(self, name_, filename, resistancesP=None, resistancesN=None, biasP=None, biasN=None, saveValues=None):
        customWeights = False
        if resistancesP is not None and resistancesN is not None and biasP is not None and biasN is not None:
            customWeights = True
        with open(filename, 'r') as f:
            data = json.load(f)["weights"]
        layers = {}
        for key in data:
            if key.startswith('fc'):
                name, t = key.split('.')
                if name not in layers:
                    layers[name] = {}
                layers[name][t] = np.array(data[key]['data'])
        layers = list(layers.values())
        self.layers = layers
        numInputs = layers[0]['weight'].shape[1]
        numOutputs = layers[-1]['weight'].shape[0]
        if customWeights:
            # assert that the custom weights have the same shape as the model
            for i in range(len(layers)):
                assert layers[i]['weight'].shape == resistancesP[i].shape, f'Custom weights for layer {i} have shape {resistancesP[i].shape}, expected {layers[i]["weight"].shape}'
                assert layers[i]['weight'].shape == resistancesN[i].shape, f'Custom weights for layer {i} have shape {resistancesN[i].shape}, expected {layers[i]["weight"].shape}'
                assert layers[i]['bias'].shape == biasP[i].shape, f'Custom bias for layer {i} have shape {biasP[i].shape}, expected {layers[i]["bias"].shape}'
                assert layers[i]['bias'].shape == biasN[i].shape, f'Custom bias for layer {i} have shape {biasN[i].shape}, expected {layers[i]["bias"].shape}'
        # print(numInputs, numOutputs)
        # create input nodes
        inputNodesNames = [f'input_{i}' for i in range(numInputs)]
        outputNodesNames = [f'output_{j}' for j in range(numOutputs)]
        __nodes__ = inputNodesNames + outputNodesNames

        if saveValues is not None:
            saveData = {}

        # print(__nodes__)
        SubCircuit.__init__(self, name_, *__nodes__)
        # create the connections between layers
        currentInputNodes = inputNodesNames.copy()
        for lnumber in range(len(layers)):
            layer = layers[lnumber]
            numLayerInputs = layer['weight'].shape[1]
            numLayerOutputs = layer['weight'].shape[0]
            if not customWeights:
                layerWeightsP, layerWeightsN = generate_P_N_resistances(1/layer['weight'])
                layerBiasP, layerBiasN = generate_P_N_bias(1/layer['bias'])
            else:
                layerWeightsP = resistancesP[lnumber]
                layerWeightsN = resistancesN[lnumber]
                layerBiasP = biasP[lnumber]
                layerBiasN = biasN[lnumber]
            if saveValues is not None:
                saveData[f'layer_{lnumber}'] = {'weightsP': layerWeightsP.tolist(), 'weightsN': layerWeightsN.tolist(), 'biasP': layerBiasP.tolist(), 'biasN': layerBiasN.tolist()}
            self.subcircuit(MemristorCrossbar(f'crossbar_{lnumber}', num_inputs=numLayerInputs, num_outputs=numLayerOutputs, resistancesP=layerWeightsP.T, resistancesN=layerWeightsN.T, biasP=layerBiasP, biasN=layerBiasN))
            layerOutputNodesNames = [f'output_{lnumber}_{j}' for j in range(numLayerOutputs)]
            if lnumber == len(layers)-1:
                # name them output nodes
                layerOutputNodesNames = outputNodesNames.copy()
            # print(currentInputNodes, layerOutputNodesNames)
            self.X(f'crossbar_{lnumber}', f'crossbar_{lnumber}', *currentInputNodes, *layerOutputNodesNames)
            currentInputNodes = layerOutputNodesNames.copy()
        if saveValues is not None:
            with open(saveValues, 'w') as f:
                json.dump(saveData, f)

if __name__ == '__main__':
    # lapiqueTest()
    # exit()
    import json
    # lapiqueConnectionTest()
    # load xor weights
    filename = 'xor_weights_snn.json'
    # filename = 'Net_model.json'
    # filename = "Net_xor.json"
    # connect inputs to the first layer
    # inputVoltages = [0, 0] # 0, 0 > 0
    # inputVoltages = [1, 0] # 1, 0 > 1
    # inputVoltages = [0, 1] # 1, 0 > 1
    # inputVoltages = [1, 1] # 1, 1 > 0
    correct = 0
    for inputVoltages in [[0, 0], [1, 0], [0, 1], [1, 1]]:
        circuit = Circuit('Xor model')
        circuit.include("uopamp_v1.1.lib")
        circuit.subcircuit(SNNModel('snn', filename))
        for i in range(len(inputVoltages)):
            circuit.V(f'input_{i}', f'input_{i}', circuit.gnd, inputVoltages[i]@u_V)
        # connect the last layer to the output
        circuit.X('snn', 'snn', 'input_0', 'input_1', 'output_0')
        # add resistance load to final output
        circuit.R(f'load', 'output_0', circuit.gnd, 1@u_kΩ)

        # print(circuit)
        
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)

        analysis = simulator.transient(step_time=1@u_ms, end_time=10@u_ms)

        # print each node name
        # for node in analysis.nodes:
        #     if node.startswith('x'):
        #         pass
        #     else:
        #         print(node, np.array(analysis[node])[-1])
        # print max voltage at each output
        # print("output voltages")
        # for j in range(1):
            # print(f'max voltage at output_{j}', np.min(np.array(analysis[f'output_{j}'])))
        maxVoltage = np.max(np.array(analysis['output_0']))
        if maxVoltage > 0.5:
            if np.sum(inputVoltages) == 1:
                correct += 1
        else:
            if np.sum(inputVoltages) == 0 or np.sum(inputVoltages) == 2:
                correct += 1
    print("The circuit obtained an accuracy of", correct*100/4, "% on the test set.")
    
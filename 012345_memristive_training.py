import numpy as np
import PIL
import json

# Now import PySpice modules
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from scipy.linalg import solve
import matplotlib.pyplot as plt
from PySpice.Unit import *
from PySpice.Probe.Plot import plot

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory

from memristor import Memristor, add_initial_condition
from snn_circuit import LapiqueNeuron

def saveVoltages(analysis, filename):
    """Save the voltages of the nodes at the end of the simulation in a json file."""
    with open(filename, 'w') as f:
        voltages = {}
        for node in analysis.nodes:
            voltages[node] = np.array(analysis[node])[-1]
        json.dump(voltages, f)

def loadVoltages(simulator, filename):
    """Load the voltages of the nodes for initial condition from a json file."""
    with open(filename, 'r') as f:
        voltages = json.load(f)
        # keep only the Cmem values
        for key in list(voltages.keys()):
            if 'cmem' not in key.lower():
                voltages.pop(key)
        simulator.initial_condition(**voltages)

def plotXPUEImages(data_X):
    """Plot the images of the XPUE dataset."""
    shapes_names = ['X', '+', 'U', '=']

    # plot them all in a single image, without the titles or axes
    fig, axs = plt.subplots(1,len(shapes_names))
    for i in range(len(data_X)):
        axs[i].imshow(np.array(data_X[i]).reshape(3, 3), cmap='gray')
        axs[i].axis('off')
        # add name of the shape
        axs[i].text(0.8, -1, shapes_names[i], fontsize=12, color='white')
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle('XPUE 3x3 images', fontsize=16, color='white')
    # set background color to black
    fig.patch.set_facecolor('black')
    plt.savefig('./xpue_dataset_images.png')
    plt.show()

class ShallowMemristorNetworkSubcircuit(SubCircuit):
    """A shallow memristor network subcircuit with a single layer of memristors and a single layer of Lapicque neurons."""
    def __init__(self, name, num_classes, num_inputs, K=0.000000005, Kp=0.13*10**-3, C=10**-1@u_pF):
        _inputNodes = [f'in{i}' for i in range(num_inputs)]
        _outputNodes = [f'out{i}' for i in range(num_classes)]
        # _controlNodes = [f'phi{j}_{i}' for j in range(num_classes) for i in range(num_inputs)]
        _controlNodes = [f'phi{j}' for j in range(num_classes)]
        __nodes__ = _inputNodes + _outputNodes + _controlNodes
        
        if len(__nodes__) > 1004:
            raise Exception(f'Too many nodes for the subcircuit: {len(__nodes__)}/1004')
        SubCircuit.__init__(self, name, *__nodes__)
        for j in range(num_classes):
            for i in range(num_inputs):
                self.subcircuit(Memristor(f'Memristor_P{j}_{i}', K, Kp, C, 0, 10**20, 10**-3))
                self.X(f'Memristor_P{j}_{i}', f'Memristor_P{j}_{i}', f'op_{j}', f'in{i}', f'phi{j}_{i}')
            self.subcircuit(LapiqueNeuron(f'neuron{j}', 1, 0.00015, 4))
            self.X(f'neuron{j}', f'neuron{j}', f'op_{j}', f'on{j}', f'out{j}')
            # add control nodes through 0 V voltages
            for i in range(num_inputs):
                self.V(f'phi_union{j}_{i}', f'phi{j}', f'phi{j}_{i}', 0@u_V)

def GenerateMemristorNetworkCircuit(data_X, slearn, K=0.000000005, Kp=0.13*10**-3, C=10**-1@u_pF, pulseDurations=20000, timeStep=100, periodDuration=80000):
    """Generate the circuit for the shallow memristor network."""
    num_inputs = len(data_X[0])
    num_classes = len(data_X)

    circuit = Circuit('symbol_memristor_snn')
    circuit.include("uopamp_v1.1.lib")

    phi_j_start_time = [periodDuration*i for i in range(num_classes)]

    for i in range(num_inputs):
        circuit.PulseVoltageSource(f'in_0_{i}', f'in_0_{i}', circuit.gnd, 0@u_V, data_X[0][i]@u_V, pulseDurations@u_us, 10000@u_ms, 0, 0, 0)
        for j in range(1, num_classes-1):
            circuit.PulseVoltageSource(f'in_{j}_{i}', f'in_{j}_{i}', f'in_{j-1}_{i}', 0@u_V, data_X[j][i]@u_V, pulseDurations@u_us, 10000@u_ms, j*periodDuration@u_us, 0, 0)
        circuit.PulseVoltageSource(f'in_{num_classes-1}_{i}', f'in{i}', f'in_{num_classes-2}_{i}', 0@u_V, data_X[num_classes-1][i]@u_V, pulseDurations@u_us, 10000@u_ms, (num_classes-1)*periodDuration@u_us, 0, 0)


    circuit.subcircuit(ShallowMemristorNetworkSubcircuit('snn', num_classes, num_inputs, K, Kp, C))
    nodes = [f'in{i}' for i in range(num_inputs)] + [f'out{i}' for i in range(num_classes)] + [f'phi{j}' for j in range(num_classes)]
    # print(nodes)
    circuit.X('snn', 'snn', *nodes)

    for j in range(num_classes):
        circuit.R(f'load{j}', f'out{j}', circuit.gnd, 1@u_kΩ)

        if slearn:
            # control voltages
            circuit.PulseVoltageSource(f'phi{j}', f'phi{j}', circuit.gnd, 0@u_V, 1.2@u_V, pulseDurations@u_us, 1000@u_ms, phi_j_start_time[j]@u_us, 0, 0)
        else:
            # 0 V control voltages
            circuit.V(f'phi{j}', f'phi{j}', circuit.gnd, 0@u_V)
    return circuit

    
def trainShallowMemristorNetwork(data_X, slearn, K=0.000000005, Kp=0.13*10**-3, C=10**-1@u_pF, pulseDurations=20000, timeStep=100, periodDuration=80000,datasetName='testDataset'):
    """Train the shallow memristor network on patterns data_X"""
    num_inputs = len(data_X[0])
    num_classes = len(data_X)
    inputSums = [sum(data_X[i]) for i in range(len(data_X))]
    inputsAverageSum = sum(inputSums)/len(inputSums)

    circuit = GenerateMemristorNetworkCircuit(data_X, slearn, K, Kp, C, pulseDurations, timeStep, periodDuration)
    print(circuit)
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    # load the initial conditions from json
    if not slearn:
        loadVoltages(simulator, f'{datasetName.lower()}_SNN_voltages.json')

    analysis = simulator.transient(step_time=timeStep@u_us, end_time=periodDuration*num_classes@u_us, use_initial_condition=True)

    if slearn:
        saveVoltages(analysis, f'{datasetName.lower()}_SNN_voltages.json')
        pass
    else:
        accuracy = ComputeAcuracy(data_X, num_classes, periodDuration, timeStep, analysis)
        print(f'Accuracy: {accuracy}')
        PlotConfusionMatrix(data_X, num_classes, periodDuration, timeStep, analysis, datasetName)
    if slearn:
        PlotMemristorX(analysis, num_inputs, num_classes, periodDuration, timeStep, datasetName)
        matrix = CapacitorVoltagesMatrix(analysis, num_inputs, num_classes)
        print(matrix)
        PlotInputVoltages(analysis, num_inputs, num_classes, periodDuration, datasetName)
        PlotControlVoltages(analysis, num_classes, num_inputs, datasetName)
    if not slearn:
        PlotOutputVoltages(analysis, num_classes, periodDuration, timeStep, datasetName)

def PlotConfusionMatrix(data_X, num_classes, periodDuration, timeStep, analysis, datasetName='testDataset'):
    """Plot the confusion matrix of the pattern classification."""
    confusionMatrix = ConfusionMatrix(data_X, num_classes, periodDuration, timeStep, analysis)
    plt.imshow(confusionMatrix, cmap='gray')
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_classes))
    plt.title(f'Confusion matrix of the {datasetName} classification')
    plt.savefig(f'./{datasetName.lower()}_confusion_matrix.png')
    plt.show()

def ConfusionMatrix(data_X, num_classes, periodDuration, timeStep, analysis):
    # split the voltages of all output nodes in intervals of pulseDurations*2
    intervals = [[np.array(analysis[node])[i*periodDuration//timeStep:(i+1)*periodDuration//timeStep] for i in range(num_classes)] for node in [f'out{j}' for j in range(num_classes)]]

    # for each interval find which output node reached 0.5V first
    confusion = np.zeros((num_classes, num_classes))
    for i in range(len(intervals)):
        for k in range(len(intervals[i][0])):
            for j in range(len(intervals[i])):
                clas = False
                if intervals[j][i][k] > 0.5:
                    print(f'Image {i} classified as {j}, time: {k} us, voltage: {intervals[j][i][k]}')
                    clas = True
                    if i == j:
                        # check if no other output node reached 0.5V at the same time
                        for l in range(len(intervals[i])):
                            if l != j and intervals[l][i][k] > 0.5:
                                print(f'Image {i} classified as {l} at the same time')
                                clas = False
                                break
                        if clas:
                            confusion[i][j] += 1
                            break
            if clas:
                break
    return confusion

def ComputeAcuracy(data_X, num_classes, periodDuration, timeStep, analysis):
    # split the voltages of all output nodes in intervals of pulseDurations*2
    intervals = [[np.array(analysis[node])[i*periodDuration//timeStep:(i+1)*periodDuration//timeStep] for i in range(num_classes)] for node in [f'out{j}' for j in range(num_classes)]]

    # for each interval find which output node reached 0.5V first
    correct = 0
    for i in range(len(intervals)):
        for k in range(len(intervals[i][0])):
            for j in range(len(intervals[i])):
                clas = False
                if intervals[j][i][k] > 0.5:
                    print(f'Image {i} classified as {j}, time: {k} us, voltage: {intervals[j][i][k]}')
                    clas = True
                    if i == j:
                        # check if no other output node reached 0.5V at the same time
                        for l in range(len(intervals[i])):
                            if l != j and intervals[l][i][k] > 0.5:
                                print(f'Image {i} classified as {l} at the same time')
                                clas = False
                                break
                        if clas:
                            correct += 1
                            break
            if clas:
                break
    return correct/len(data_X)

def PlotMemristorX(analysis, num_inputs, num_classes, periodDuration, timeStep, datasetName='testDataset'):
    # plot the voltages at node cmem nodes in a single plot
    fig, axs = plt.subplots(num_inputs)
    for i in range(num_inputs):
        for j in range(num_classes):
            axs[i].plot(np.array(analysis.nodes[f'xsnn.xmemristor_p{j}_{i}.cmem']), label=f'X_{j}_{i}')
        # remove x ticks
        axs[i].set_xticks([])
        maxValue = round(max([max(np.array(analysis.nodes[f'xsnn.xmemristor_p{j}_{i}.cmem'])) for j in range(num_classes)]))
        axs[i].set_yticks([0, maxValue])
        axs[i].set_ylim(0, 1.2*maxValue)
    fig.set_size_inches(10, 10)
    fig.legend(['j=0', 'j=1', 'j=2', 'j=3'], loc='right')
    for i in range(num_inputs):
        fig.text(0.05, 0.855-(0.78/(num_inputs))*i, f'X_{i}', fontsize=10, color='black')
    # add x ticks
    _begin = 0
    _end = periodDuration*num_classes/timeStep+1
    _step = periodDuration/timeStep
    print("Begin: ", _begin, "End: ", _end, "Step: ", _step)
    plt.xticks(np.arange(_begin, _end, _step))
    plt.xlabel('Time step')
    plt.suptitle(f'Memristor capacitor voltages (X(t)) for the \'{datasetName}\' dataset')
    plt.savefig(f'./{datasetName.lower()}_capacitor_voltages.png')
    plt.show()
    plt.clf()

def CapacitorVoltagesMatrix(analysis, num_inputs, num_classes):
    matrix = np.zeros((num_inputs, num_classes))
    for i in range(num_inputs):
        for j in range(num_classes):
            matrix[i][j] = np.array(analysis.nodes[f'xsnn.xmemristor_p{j}_{i}.cmem'])[-1]
    return matrix

def PlotInputVoltages(analysis, num_inputs, num_classes, periodDuration, datasetName='testDataset'):
    # plot the input voltages in a single figure, vertically
    fig, axs = plt.subplots(num_inputs, 1)
    for j in range(num_inputs):
        # plot with different colors
        color = plt.cm.viridis(j/num_inputs)
        axs[j].plot(np.array(analysis.nodes[f'in{j}']), label=f'V{j}', c=color)
        # remove y ticks
        axs[j].set_yticks([])
        # remove x ticks
        axs[j].set_xticks([])
        # set the y interval to 0,1
        axs[j].set_ylim(0, 1)
        axs[j].legend([f'V_{j}'], fontsize=4)
    # set the title of the plot
    fig.suptitle(f'Input data voltages for the \'{datasetName}\' dataset')
    # set plot size to be larger
    fig.set_size_inches(10, 10)
    plt.xlabel('Time step')
    plt.xticks(np.arange(0, num_classes*periodDuration/timeStep+1, pulseDuration/timeStep))
    # add texts
    for i in range(num_inputs):
        fig.text(0.05, 0.855-(0.78/(num_inputs))*i, f'V_{i}', fontsize=10, color='black')
    plt.suptitle(f'Input voltages for the \'{datasetName}\' dataset')
    plt.savefig(f'./{datasetName.lower()}_input_voltages.png')
    plt.show()
    plt.clf()

def PlotControlVoltages(analysis, num_classes, num_inputs, datasetName='testDataset'):
    # plot the control voltages
    fig, axs = plt.subplots(num_classes, 1)

    # add plots of the phi voltages
    for j in range(num_classes):
        try:
            color = plt.cm.viridis(j/num_classes)
            axs[j].plot(np.array(analysis.nodes[f'phi{j}']), label=f'phi{j}', c=color)
            axs[j].set_yticks([])
            axs[j].legend([f'phi_{j}'], fontsize=4)
            axs[j].set_ylim(0, 1.2)
        except Exception as e:
            print(e)
            print(f'phi{j} not found')
    # set the title of the plot
    fig.suptitle(f'Control voltages for the \'{datasetName}\' dataset')
    # set plot size to be larger
    fig.set_size_inches(10, 10)
    plt.xlabel('Time step')
    plt.suptitle(f'Control voltages for the \'{datasetName}\' dataset')
    # add texts
    for i in range(num_classes):
        fig.text(0.05, 0.855-(0.78/(num_classes))*i, f'phi_i_{i}', fontsize=10, color='black')
    plt.savefig(f'./{datasetName.lower()}_control_voltages.png')
    plt.show()
    plt.clf()

def PlotOpNodeVoltages(analysis, num_classes, num_inputs):
    # plot the op_j node voltages and then the input voltages
    fig, axs = plt.subplots(2)
    for j in range(num_classes):
        axs[0].plot(np.array(analysis.nodes[f'op_{j}']), label=f'op{j}')
    for i in range(num_inputs):
        axs[1].plot(np.array(analysis.nodes[f'in{i}']), label=f'in{i}')
    plt.legend()
    plt.show()

def PlotOutputVoltages(analysis, num_classes, periodDuration, timeStep, datasetName='testDataset'):
    # output voltages
    fig, axs = plt.subplots(num_classes, 1)
    # set the size of the figure
    fig.set_size_inches(10, 10)
    for i in range(num_classes):
        for j in range(num_classes):
            axs[i].plot(np.array(analysis.nodes[f'out{j}'][i*periodDuration//timeStep:i*periodDuration//timeStep+periodDuration//(6*timeStep)]), label=f'out{j}')
        axs[i].set_yticks([])
        # set the y interval to 0,1
        axs[i].set_ylim(0, 1)
        # axs[i].text(-29, 0.4, f'Input image {i}', fontsize=12, color='black')
    fig.suptitle(f'Output neuron voltages for the \'{datasetName}\' dataset')
    fig.legend([f"out_{j}" for j in range(num_classes)], loc='right')
    plt.xlabel('Time step')
    # add texts
    for i in range(num_classes):
        fig.text(0.05, 0.855-(0.78/(num_classes))*i, f'out_{i}', fontsize=10, color='black')
    plt.savefig(f'./{datasetName.lower()}_output_voltages.png')
    plt.show()


def plot012345Images(data_X):
    # plot in a single plot each 3x5 image as for xpue dataset
    fig, axs = plt.subplots(1, 6)
    for i in range(6):
        axs[i].imshow(np.array(data_X[i]).reshape(5, 3), cmap='gray')
        axs[i].axis('off')
        axs[i].text(0.8, -1, str(i), fontsize=12, color='white')
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle('012345 images', fontsize=16, color='white')
    # set background color to black
    fig.patch.set_facecolor('black')
    plt.savefig('./012345_dataset_images.png')
    plt.show()

if __name__ == "__main__":
    data_X = [
        [1,1,1,1,0,1,1,0,1,1,0,1,1,1,1],
        [0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
        [1,1,1,0,0,1,1,1,1,1,0,0,1,1,1],
        [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1],
        [1,0,1,1,0,1,1,1,1,0,0,1,0,0,1],
        [1,1,1,1,0,0,1,1,1,0,0,1,1,1,1],
    ]
    plot012345Images(data_X)

    timeStep = 10
    pulseDuration = 5000
    periodDuration = 3*pulseDuration
    name='012345'

    slearn = True

    trainShallowMemristorNetwork(data_X, slearn, K=0.000000005, Kp=0.13*10**-3, C=10**-1@u_pF, pulseDurations=pulseDuration, timeStep=timeStep, periodDuration=periodDuration, datasetName=name.upper())

import numpy as np
import PIL
import json

# Now import PySpice modules
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from scipy.linalg import solve
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PySpice.Unit import *
from PySpice.Probe.Plot import plot

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from memristive_training import ShallowMemristorNetworkSubcircuit, loadVoltages

data_path='/tmp/data/mnist'

def loadMNIST():
    transform = transforms.Compose([
                # transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    return mnist_train, mnist_test


train_loader, test_loader = loadMNIST()
avg_images = []
for j in range(10):
    # visualize average image of class j
    # j = 3
    sum_images_j = 0
    count = 0
    for image, label in train_loader:
        if label == j:
            count += 1
            sum_images_j += image
        if count == 100:
            break
    avg_image_j = sum_images_j / count
    # print(avg_image_j.shape)
    # plt.imshow(avg_image_j[0], cmap='gray')
    # plt.show()
    avg_images.append(avg_image_j)

# resize the images to sizexsize
size=10
data_X = [np.array(avg_images[i][0]) for i in range(10)]
data_X = np.array([PIL.Image.fromarray((data_X[i] * 255).astype(np.uint8)).resize((size, size)) for i in range(10)])

# keep only the 10 greatest pixel values (set 1) and the rest to 0
data_X = np.array([np.array(data_X[i]) > np.sort(data_X[i].flatten())[-20] for i in range(10)])

data_X = np.array([data_X[i].flatten() for i in range(10)])
num_inputs = len(data_X[0])
num_classes = 10

timeStep = 11
pulseDurations = 2000
periodDuration = 3*pulseDurations
name = "mnist"

K=0.000000005
Kp=0.13*10**-3
C=10**-1@u_pF

correct = 0
top3_correct = 0
total = 0
confusion_matrix = np.zeros((num_classes, num_classes))
# for each image in the test set, calculate the output of the network
for k, (image, label) in enumerate(test_loader):
    # resize image to 10x10
    image = PIL.Image.fromarray((image[0].numpy() * 255).astype(np.uint8)).resize((size, size))
    # set values to 0 or 1
    image = np.array(image) > np.sort(np.array(image).flatten())[-20]
    image = np.array(image).astype(int).flatten()
    
    circuit = Circuit('mnist_memristor_inference')
    circuit.include("uopamp_v1.1.lib")

    for i in range(num_inputs):
        circuit.V(f'in{i}', f'in{i}', circuit.gnd, image[i]@u_V)
    
    circuit.subcircuit(ShallowMemristorNetworkSubcircuit('snn', num_classes, num_inputs, K, Kp, C))
    nodes = [f'in{i}' for i in range(num_inputs)] + [f'out{i}' for i in range(num_classes)] + [f'phi{j}' for j in range(num_classes)]
    # print(nodes)
    circuit.X('snn', 'snn', *nodes)
    for j in range(num_classes):
        circuit.R(f'load{j}', f'out{j}', circuit.gnd, 1@u_kΩ)
        circuit.V(f'phi{j}', f'phi{j}', circuit.gnd, 0@u_V)
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    loadVoltages(simulator, 'mnist_SNN_voltages.json')

    analysis = simulator.transient(step_time=timeStep@u_us, end_time=periodDuration@u_us, use_initial_condition=True)

    # find which output spiked first
    output = np.array([analysis.nodes[f'out{i}'] for i in range(num_classes)])
    print(output)
    # find the first spike
    firstSpike = np.argmax(output > 0.5, axis=1)
    # cast it as float
    firstSpike = firstSpike.astype(float)
    spikesValues = np.max(output, axis=1)
    # set as np.inf the values that did not spike
    firstSpike[spikesValues < 0.5] = np.inf
    # find the class with the first spike
    selectedClass = np.argmin(firstSpike)
    print(f'Predicted class: {selectedClass}, True class: {label}')
    confusion_matrix[label, selectedClass] += 1
    if selectedClass == label:
        correct += 1
    total += 1
     # top 3 accuracy
    top3 = np.argsort(firstSpike)[:3]
    if label in top3:
        top3_correct += 1
    print(f'Index: {k}, Correct: {correct}, Total: {total}')
    print(f'Accuracy: {100*correct/total}%')
    print(f'Top 3 accuracy: {100*top3_correct/total}%')
    if k == 100:
        break
print(f'Index: {k}, Correct: {correct}, Total: {total}')
print(f'Accuracy: {100*correct/total}%')
print(f'Top 3 accuracy: {100*top3_correct/total}%')

# plot confusion matrix
fig, ax = plt.subplots(1,1)
im = ax.imshow(confusion_matrix)
# We want to show all ticks...
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
# ... and label them with the respective list entries
ax.set_xticklabels(np.arange(num_classes))

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=15, ha="right",
        rotation_mode="anchor")
plt.xlabel('Predicted class')
plt.ylabel('True class')

# Loop over data dimensions and create text annotations.
for i in range(num_classes):
    for j in range(num_classes):
        text = ax.text(j, i, confusion_matrix[i, j],
                    ha="center", va="center", color="w")
        
ax.set_title("Confusion matrix for MNIST memristor inference")
fig.tight_layout()
plt.savefig('./mnist_memristor_inference_confusion_matrix.png')
plt.show()

# In-Situ-Memristive-Learning-Method-for-Pattern-Classification

After proposing the SNNtrainer3D app (https://github.com/jurjsorinliviu/SNNtrainer3D and https://www.mdpi.com/2076-3417/14/13/5752), and the Efficient Implementation of Spiking Neural Networks for Inference using Ex-Situ Training (https://github.com/jurjsorinliviu/SNNs_for_Inference_using_Ex-Situ_Training and https://ieeexplore.ieee.org/document/10681427), here, a memristive learning method is proposed called "In-Situ Memristive Learning Method for Pattern Classification".

**NEW**: The paper behind this implementation called "PySpice-Simulated In Situ Learning with Memristor Emulation for Single-Layer Spiking Neural Networks
" is published here: https://www.mdpi.com/2079-9292/13/23/4665

To summarize, in this work, I trained SNNs completely in-situ using memristor emulators in SPICE that can be parametrized to emulate any particular memristor. Also, I developed the first PySpice implementation of memristors. I trained a few datasets with this method with great results, including MNIST. Regarding possible future works, some ideas could be built on top of this: backpropagation of errors, generalizing the training method for many layers, a more technical version of STDP learning, and other datasets.

Details about implementation files:
- memristor.py defines the PySpice memristor emulator used by the other scripts, and running it by itself generates the IV plot for the simulated memristor.
- memristor_figures.py generates the theoretical memristor IV plot.
- The training files (XPUE_memristive_training.py, 012345_memristive_training.py, MNIST_memristive_training.py) generate the memristive network, trains it, saves the obtained voltages for inference later and generates some plots.
- Then the inference files (XPUE_memristive_inference.py, 012345_memristive_inference.py, MNIST_memristive_inference.py) load the respective trained voltages and perform inference to obtain train accuracy and generate some plots.
- MNIST_memristor_test.py is used to calculate the test accuracy for the MNIST circuit.
- The remaining scripts are dependencies for these main scripts.

## Citation
If you are interested in citing this work, please use the following citation:

Jurj, S.L. PySpice-Simulated In Situ Learning with Memristor Emulation for Single-Layer Spiking Neural Networks. Electronics 2024, 13, 4665. https://doi.org/10.3390/electronics13234665.

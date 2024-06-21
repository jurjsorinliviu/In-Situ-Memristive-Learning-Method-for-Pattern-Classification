import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 100, 1000)
v = np.sin(t)
c1 = 0
i = (np.exp(-np.cos(t)+c1))*np.sin(t)

plt.plot(v, i, label='V-I')
plt.title('Memristor IV curve')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.savefig('linear_memristor.png')
plt.show()
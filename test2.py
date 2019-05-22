import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt

noise = np.random.normal(0,0.3, 10000)
plt.plot(noise)
plt.show()
np.savetxt("in.txt", noise)
response = np.loadtxt("out.txt")
print(response)
plt.plot(response)
plt.show()

# delay estimation
min_theta = 0
max_theta = 30000

for theta in range(max_theta):

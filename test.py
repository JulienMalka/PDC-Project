import numpy as np
from scipy.fftpack import fft
from scipy import signal
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(1000.0 * 2.0*np.pi*x)
#y = np.loadtxt("out.txt")
y = signal.unit_impulse(600, 300)
y = np.loadtxt("out.txt")
yf = fft(y)
xf = np.linspace(0.0, 9000, N//2)

import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

np.savetxt("in.txt", y)




import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import random
from root_raised import raised_cosine



msg = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]


msg = [-1 if elem == 0 else 1 for elem in msg]

_, signal = raised_cosine(0, 5, 22050)

msg_pam = np.array([])

print(msg_pam.shape)
print(signal.shape)

for elem in msg:
    msg_pam=np.concatenate([msg_pam, elem*signal])


plt.plot(msg_pam)
plt.show()

Fs = 22050.0
Ts = 1.0/Fs


n = len(msg_pam) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

Y = np.fft.fft(msg_pam)/n # fft computing and normalization
Y = Y[:(n//2)]


plt.plot(frq, np.power(np.abs(Y), 2))
plt.show()


noise = np.random.normal(0, 0.05, len(msg_pam))

msg_noise = msg_pam + noise

plt.plot(msg_noise)
plt.show()

matched_filter = scipy.signal.correlate(msg_noise, signal)

print(len(matched_filter))

plt.plot(matched_filter/10000)
plt.show()

for i in range(1, 7):
    value = matched_filter[5*22050*i]/10000
    diff1 = np.abs(1-value)
    diff2 = np.abs(-1-value)
    if diff1<diff2:
        print(1)
    else:
        print(0)
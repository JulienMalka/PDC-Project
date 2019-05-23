import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import random
from root_raised import raised_cosine

train_sig = [-1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1,
             -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1,
             1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1,
             1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1,
             -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1,
             -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1,
             1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1,
             -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
             -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1,
             1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1,
             -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1,
             -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1,
             1, 1, -1, 1]


len_train = len(train_sig)

train_sig = np.array(train_sig)

info = msg = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

msg = [-1 if elem == 0 else 1 for elem in msg]

_, signal = raised_cosine(0, 1, 22050)


msg_sig = train_sig



for elem in msg:
    msg_sig=np.concatenate([msg_sig, elem*signal])



pulse = np.linspace(0, 20, 441300)
msg_sig = msg_sig * np.exp(1j * 20000 * pulse)
msg_sig = msg_sig * np.exp(1j * -20000 * pulse)




plt.plot(msg_sig.real)
plt.show()
np.savetxt("in_serv.txt", msg)


Fs = 22050.0
Ts = 1.0/Fs
n = len(msg_sig)
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range
Y = np.fft.fft(msg_sig)/len(msg_sig) # fft computing and normalization
Y = Y[range(int(n/2))]
plt.plot(frq, np.abs(Y))
plt.show()



signal_recovered = np.fft.ifft(Y).real
ls = len(signal_recovered)
#plt.plot(np.concatenate([signal_recovered[ls//2:], signal_recovered[:ls//2]]))
plt.plot(signal_recovered)
plt.show()


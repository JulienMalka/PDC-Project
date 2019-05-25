import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import random
from scipy.signal import butter, lfilter, freqz

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



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


len_train = len(train_sig)
train_sig = np.array(train_sig)

response  = np.loadtxt("out_serv.txt")
msg = np.loadtxt("in_serv_save.txt")
msg = msg[len_train:]

estimated_delay = np.argmax(scipy.signal.correlate(response, train_sig)) - len_train + 1

print(f"The estimated delay is {estimated_delay}")
print(f"Actual delay is {response.shape[0]-msg.shape[0]}")

reconstructed_msg = response[estimated_delay+len_train:]
plt.plot(reconstructed_msg)
plt.show()




pulse = np.linspace(0, 189640/22050, 189640)
reconstructed_msg = reconstructed_msg * np.exp(1j * -7900*2*np.pi * pulse)

reconstructed_msg = butter_lowpass_filter(reconstructed_msg, 500, 22050, order=15)


ft = fft(reconstructed_msg)
ft2 = fft(msg)


Fs = 22050.0
Ts = 1.0/Fs # sampling interval
t = np.arange(0,1,Ts) # time vector

n = len(reconstructed_msg) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

Y = np.fft.fft(reconstructed_msg)/n # fft computing and normalization
Y = Y[:(n//2)]

fig, ax = plt.subplots(2, 1)
ax[0].plot(reconstructed_msg.real)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plt.show()
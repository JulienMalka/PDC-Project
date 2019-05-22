import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import random

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

response  = np.loadtxt("out_serv.txt")
msg = np.loadtxt("in_serv_save.txt")
msg = msg[len_train:]

estimated_delay = np.argmax(scipy.signal.correlate(response, train_sig)) - len_train + 1

print(f"The estimated delay is {estimated_delay}")
print(f"Actual delay is {response.shape[0]-msg.shape[0]}")

reconstructed_msg = response[estimated_delay+len_train:estimated_delay+len_train+10000]
plt.plot(reconstructed_msg)
plt.show()


ft = fft(reconstructed_msg)
ft2 = fft(msg)
plt.plot(np.abs(ft))
plt.show()
plt.plot(np.abs(ft2))
plt.show()
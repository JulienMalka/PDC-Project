import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import random

train_sig =

len_train = len(train_sig)

train_sig = np.array(train_sig)

info = np.random.normal(0,0.3, 10000)

msg = np.concatenate([train_sig, info])
plt.plot(msg)
plt.show()



delay = random.randint(2000, 20000)
print(f"The delay is {delay}")

noise_delay = np.random.normal(0, 0.05, delay)
noise_msg = np.random.normal(0, 0.05, msg.shape[0])

noised_msg = msg + noise_msg

delayed_msg = np.concatenate([noise_delay, noised_msg])

plt.plot(delayed_msg)
plt.show()



estimated_delay = np.argmax(scipy.signal.correlate(delayed_msg, train_sig)) - len_train + 1

print(f"The estimated delay is {estimated_delay}")

plt.plot(delayed_msg[estimated_delay+len_train:])
plt.show()
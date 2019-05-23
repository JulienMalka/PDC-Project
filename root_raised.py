import commpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft


def raised_cosine(shifted_freq, time_per_symb, samp_freq, roll_off=0.5):
    shifted_freq /= 2

    pulse = np.linspace(-time_per_symb, time_per_symb, samp_freq * time_per_symb)

    numerator = np.cos((1 + roll_off) * np.pi * pulse / time_per_symb) + \
                (1 - roll_off) * np.pi / (4 * roll_off) * np.sinc((1 - roll_off) * pulse / time_per_symb)
    denominator = 1 - (4 * roll_off * pulse / time_per_symb) ** 2

    limit_case = roll_off / (np.pi * np.sqrt(2 * time_per_symb)) * ((np.pi + 2) * np.sin(np.pi / (4 * roll_off)) + \
                                                                    (np.pi - 2) * np.cos(np.pi / (4 * roll_off)))

    out = 4 * roll_off / (np.pi * np.sqrt(time_per_symb)) * numerator / denominator
    #out = numerator/denominator
    out[np.isnan(out)] = limit_case
    #out *= np.cos(2 * np.pi * shifted_freq * pulse)
    return pulse, out



pulse, test = raised_cosine(1000, 1, 22050)
Fs = 22050.0
Ts = 1.0/Fs

plt.plot(pulse, test)
plt.show()

n = len(test) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
#frq = frq[range(int(n/2))] # one side frequency range

Y = np.fft.fft(test)/n # fft computing and normalization




plt.plot(frq, np.power(np.abs(Y), 2))
plt.show()
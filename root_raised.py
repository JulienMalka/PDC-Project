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
    #out[np.isnan(out)] = limit_case
    out *= np.cos(2 * np.pi * shifted_freq * pulse)
    return pulse, out



# pulse, signal1 = raised_cosine(2000, 1, 22050)
# pulse, signal2 = raised_cosine(4000, 1, 22050)
# pulse, signal3 = raised_cosine(6000, 1, 22050)
# pulse, signal4 = raised_cosine(8000, 1, 22050)
#
# signal_fin = signal1 + signal2 + signal3 + signal4
#
#
#
# Fs = 22050.0
# Ts = 1.0/Fs
#
# plt.plot(pulse, signal_fin)
# plt.show()
#
# n = len(signal1) # length of the signal
# k = np.arange(n)
# T = n/Fs
# frq = k/T # two sides frequency range
# frq = frq[range(int(n/2))] # one side frequency range
#
# Y = np.fft.fft(signal1)/len(signal1) # fft computing and normalization
# Y = Y[range(int(n/2))]
#
#
# Y = np.abs(Y)
# plt.plot(np.abs(Y))
# plt.show()
#
#
# n = len(signal_fin)
# k = np.arange(n)
# T = n/Fs
# frq = k/T # two sides frequency range
# frq = frq[range(int(n/2))] # one side frequency range
#
#
# Y = np.fft.fft(signal_fin)/len(signal_fin) # fft computing and normalization
# Y = Y[range(int(n/2))]
#
#
# Y = np.abs(Y)
# y_len = len(Y)
# Y = Y[2003:4005]
# Y = np.concatenate([Y, np.zeros(y_len-len(Y))])
#
# plt.plot(np.abs(Y))
# plt.show()
#
# signal_recovered = np.fft.ifft(Y).real
# ls = len(signal_recovered)
# plt.plot(np.concatenate([signal_recovered[ls//2:], signal_recovered[:ls//2]]))
# plt.show()
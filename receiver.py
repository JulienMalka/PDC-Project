import matplotlib.pyplot as plt
import scipy
from helpers import *


print("Loading training signals...")
len_train, train_sig_start = generate_training_sig_start()
_, train_sig_end = generate_training_sig_end()


print("Loading response from server...")
response = np.loadtxt("out_serv.txt")


print("Estimating delays...")

estimated_delay_start = np.argmax(scipy.signal.correlate(response, train_sig_start)) - len_train + 1
estimated_delay_end = np.argmax(scipy.signal.correlate(response, train_sig_end)) - len_train + 1

print("Reconstructing signal...")
reconstructed_msg = response[estimated_delay_start+len_train:estimated_delay_end]


print("Estimating cut frequencies interval...")
frq_reconstructed_msg, Y_reconstructed_msg = compute_fft(reconstructed_msg)
i = np.where(frq_reconstructed_msg >= 1.8*1000)[0][0]
j = np.where(frq_reconstructed_msg >= 2.2*1000)[0][0]
k = np.where(frq_reconstructed_msg >= 7.8*1000)[0][0]
l = np.where(frq_reconstructed_msg >= 8.2*1000)[0][0]

sum_low = np.sum(np.abs(Y_reconstructed_msg[i:j]))
sum_high = np.sum(np.abs(Y_reconstructed_msg[k:l]))

cut = 2000 if sum_low > sum_high else 8000


print("Demodulating signal... \n")
pulse = np.linspace(0, len(reconstructed_msg)/sampling_rate, len(reconstructed_msg))
demodulated_signal = reconstructed_msg * np.exp(1j * -cut*2*np.pi * pulse)
demodulated_signal = butter_lowpass_filter(demodulated_signal, 1000, sampling_rate, order=15)


frq_demodulated, Y_demodulated = compute_fft(demodulated_signal)

_, signal = raised_cosine(period_symbol, sampling_rate)

matched_filter = scipy.signal.correlate(demodulated_signal.real, signal)

txt = ""
for i in range(1, int(len(demodulated_signal)/(sampling_rate*period_symbol)) +1):
    value = matched_filter[int(sampling_rate*i*period_symbol)]
    diff1 = np.abs(1-value)
    diff2 = np.abs(-1-value)
    if diff1<diff2:
        txt += "1"
    else:
        txt += "0"


print("==============RECOVERED FILE====================\n")

n = int(txt, 2)
recovered = n.to_bytes((n.bit_length() + 7) // 8, 'big').decode("unicode_escape")
print(recovered + "\n")
original = open("in.txt").read()
errors = 0
for i in range(len(original)):
    if original[i] != recovered[i]:
        errors+=1

print(f"Nb of errors : {errors}")



fig, ax = plt.subplots(3, 2)
fig.set_size_inches(18.5, 10.5)
ax[0][0].plot(response)
ax[0][0].set_title("Signal sent back by server")
ax[0][0].set_xlabel('Time')
ax[0][0].set_ylabel('Amplitude')

ax[0][1].plot(reconstructed_msg)
ax[0][1].set_title("Reconstructed signal after removing delays")
ax[0][1].set_xlabel('Time')
ax[0][1].set_ylabel('Amplitude')

ax[1][0].plot(frq_reconstructed_msg, np.abs(Y_reconstructed_msg), "r")
ax[1][0].set_title("Fourier transform of the reconstructed signal")
ax[1][0].set_xlabel('Freq (Hz)')
ax[1][0].set_ylabel('|Y(freq)|')

ax[1][1].plot(demodulated_signal.real)
ax[1][1].set_title("Signal after demodulation")
ax[1][1].set_xlabel('Time')
ax[1][1].set_ylabel('Amplitude')

ax[2][0].plot(frq_demodulated, np.abs(Y_demodulated), "r")
ax[2][0].set_title("Fourier transform of the demodulated signal")
ax[2][0].set_xlabel('Freq (Hz)')
ax[2][0].set_ylabel('|Y(freq)|')

ax[2][1].plot(matched_filter.real)
ax[2][1].set_title("Signal after matched filter")
ax[2][1].set_xlabel('Time')
ax[2][1].set_ylabel('Amplitude')
plt.show()
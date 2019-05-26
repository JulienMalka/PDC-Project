import numpy as np
import random
import matplotlib.pyplot as plt
from helpers import *

msg_sent = np.loadtxt("in_serv.txt")
range_deleted = random.randint(1,4)

Fs = 22050.0
Ts = 1.0/Fs # sampling interval
t = np.arange(0,1,Ts) # time vector
n = len(msg_sent) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range

Y = np.fft.fft(msg_sent)/n # fft computing and normalization



len_four = len(Y)

print(len_four)

new_Y = np.ones([len_four,])
down = 1
up = 3

i, = np.where(frq>=down*1000)
j, = np.where(frq>=up*1000)



new_Y[i[0]:j[0]] = 0

i, = np.where(frq>= (22-up)*1000 )
j, = np.where(frq>= (22-down)*1000 )
new_Y[i[0]:j[0]] = 0


Y = Y * new_Y





final_sig = np.fft.ifft(Y)*800000
final_sig = final_sig.real

delay = random.randint(2000, 20000)
print(f"The delay is {delay}")
plt.plot(final_sig.real)
plt.show()
delay2 = random.randint(2000, 50000)

noise_delay = np.random.normal(0, 0.05, delay)
noise_msg = np.random.normal(0, 0.05, final_sig.shape[0])
noise_delay2 = np.random.normal(0, 0.05, delay2)
noised_msg = final_sig + noise_msg

delayed_msg = np.concatenate([noise_delay, noised_msg, noise_delay2])



delayed_msg = [1 if elem>1 else -1 if elem<-1 else elem for elem in delayed_msg]




np.savetxt("out_serv.txt", delayed_msg)
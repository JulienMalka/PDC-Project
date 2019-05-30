from helpers import *

print("Generating training signals...")
len_train, train_sig_start = generate_training_sig_start()
_, train_sig_end = generate_training_sig_end()

print("Reading text file...")
file = open("in.txt").read()
msg = [int(bit) for elem in file for bit in format(ord(elem), "08b")]


nb_bits = len(msg)

msg = [-1 if elem == 0 else 1 for elem in msg]


print("Generating raised cosine pulse...")
_, signal = raised_cosine(period_symbol, sampling_rate)
print("Generating signal...")
msg_sig = np.array([])
for elem in msg:
    msg_sig = np.concatenate([msg_sig, elem*signal])


frq, Y = compute_fft(msg_sig)


print("Modulating frequencies of the signal...")
pulse = np.linspace(0, nb_bits*period_symbol, len(msg_sig))
msg_sig1 = msg_sig * np.exp(1j * 2000*2*np.pi * pulse)
msg_sig2 = msg_sig * np.exp(1j * 8000*2*np.pi * pulse)
msg_sig = msg_sig1+msg_sig2
msg_sig /= 2

print("Adding training signals...")
msg_sig = np.concatenate([train_sig_start, msg_sig, train_sig_end])

np.savetxt("in_serv.txt", msg_sig.real)

print("Done!")
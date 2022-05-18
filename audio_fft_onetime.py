from functions import read_wav,add_zeros
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


sound_array, fs = read_wav('D:\\Audio_files\\examples_samples_guitar.wav', mmap=False)
n, sound_array = add_zeros(sound_array, degree=12)  # where n = 2^degree
#spectrum_number = 13
sound_array = sound_array[0: fs*30]

duration = len(sound_array) / fs  # -> duration of the whole audio.
dt = 1 / fs
time_vector = np.arange(0, duration, dt)
nn = len(time_vector)




plt.figure(1)
plt.subplot(311)
plt.plot(time_vector, sound_array, color='c', label="Noisy")
plt.subplot(312)
plt.plot(freq, PSD)
plt.xlim(0, max(freq/2))
plt.subplot(313)
plt.plot(time_vector, ffilt)
plt.show()


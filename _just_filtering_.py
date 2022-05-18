# Doing FFT and IFFT (without any filter)

from functions import read_wav, add_zeros
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
sound_array, fs = read_wav('D:\\Audio_files\\sound_with_noise.wav', mmap=True)
n, sound_array = add_zeros(sound_array, degree=12)  # where n = 2^degree

duration = len(sound_array) / fs  # -> duration of the whole audio.
dt = 1 / fs
time_vector = np.arange(0, duration, dt)

f = f + noise
degree = 10
fft_points = 2 ** degree
# slicing Sound and Time arrays

sliced_sound_array = []
sliced_time_array = []
i = 0
while i < len(sound_array):
    sliced_sound_array.append(sound_array[i: i + fft_points])
    sliced_time_array.append(time_vector[i: i + fft_points])
    i += fft_points

# doing fft

PSD = []
for j in range(len(sliced_sound_array)):
    PSD.append(np.fft.fft(sliced_sound_array[j]))

freq = (1/(dt*fft_points)) * np.arange(fft_points)

filtered_powers = deepcopy(PSD)

filt_list = []
for j in range(len(PSD)):
    ffilt = np.fft.ifft(PSD[j])
    filt_list.append(ffilt)


plt.figure(1)
plt.subplot(311)
plt.plot(time_vector[:-1], sound_array,label='clean')
plt.plot(time_vector[:-1], f, label='noisy')
plt.legend()

plt.subplot(312)
for j in range(fft_points):
    plt.plot(freq, abs(PSD[j]))
    plt.xlim(0, max(freq)/2)

plt.subplot(313)
p=0
while p < len(sliced_time_array):
    plt.plot(sliced_time_array[p:p+fft_points], filt_list[p:p+fft_points], color='c')
    p += fft_points

plt.show()

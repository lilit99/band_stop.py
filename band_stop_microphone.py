from functions import read_wav, add_zeros, band_stop,fft_fft, band_stop_with_list
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io.wavfile import write

filename = 'D:\\Audio_files\\Microphone_test.wav'
#spectrum_num = 13

sound_array, fs = read_wav(filename, mmap=False)
sound_array = sound_array[0: fs*8]

n, sound_array = add_zeros(sound_array, degree=12)  # where n = 2^degree

duration = len(sound_array) / fs  # -> duration of the whole audio.
dt = 1 / fs
time_vector = np.arange(0, duration, dt)
#clean = deepcopy(sound_array)
#sound_array = sound_array + 30000*np.sin(2*np.pi*3000*time_vector) + 22000*np.sin(2*np.pi*4000*time_vector) + 25000*np.sin(2*np.pi*2000*time_vector)+ 25000*np.sin(2*np.pi*1000*time_vector)
degree = 10
fft_points = 2 ** degree
#L = np.arange(1, np.floor(fft_points/2), dtype='int')
#scaledd = np.int16(sound_array / np.max(np.abs(sound_array)) * 32767)
#write("D:\\Audio_files\\noisy.wav", fs, scaledd)
# slicing Sound and Time arrays

PSD = fft_fft(sound_array=sound_array, time_array=time_vector, fft_points=fft_points)

freq = (1/(dt*fft_points)) * np.arange(fft_points)
d_freq = freq[-1] / len(freq)


y_filt_1 = band_stop(lowcut=1470, highcut=1530, order=4, fs=fs, signal_array=sound_array)
y_filt_2 = band_stop(lowcut=1970, highcut=2030, order=4, fs=fs, signal_array=y_filt_1)

PSD2 = fft_fft(sound_array=y_filt_2, time_array=time_vector, fft_points=fft_points)
#y_filt2 = 10*y_filt2
#scaled2 = np.int16(abs(y_filt2) / np.max(np.abs(y_filt2)) * 32767)
write("D:\\Audio_files\\filtered_microphone.wav", fs, y_filt_2.astype(np.int16))
#write("D:\\Audio_files\\vochmiban.wav", fs, sound_array.astype(np.int16))


plt.figure(1)
plt.subplot(411)
plt.plot(time_vector, sound_array, label='clean')
plt.legend()

plt.subplot(412)
for j in range(len(PSD)):
    plt.plot(freq, PSD[j])
    plt.xlim(0, max(freq)/2)

plt.legend()
plt.subplot(413)
plt.plot(time_vector, y_filt_2, label='filtered')
#plt.ylim(-280, 350)
#plt.subplot(414)
#plt.plot(time_vector, y_filt2)
plt.legend()
plt.figure(2)
plt.subplot(211)
for j in range(len(PSD2)):
    plt.plot(freq,PSD2[j])
    plt.xlim(0, max(freq)/2)

plt.subplot(212)
for j in range(len(PSD)):
    plt.plot(freq, PSD[j])
    plt.xlim(0, max(freq)/2)

plt.figure(3)
for j in range(len(PSD2)):
    plt.plot(freq, PSD2[j])
    plt.xlim(0, max(freq)/2)

summ = []
for j in range(len(PSD[j])):
    nv = []
    for i in range(len(PSD)):
        nv.append(PSD[i][j])
    summ.append(sum(nv))
plt.figure(4)
plt.plot(freq, summ)
plt.xlim(0,max(freq)/2)
plt.show()
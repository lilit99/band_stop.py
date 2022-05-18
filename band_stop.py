from functions import read_wav, add_zeros, band_stop, fft_fft, band_stop_with_list
from numpy import arange, int16, sin, cos, pi, max, abs
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io.wavfile import write

filename = 'D:\\Audio_files\\examples_samples_guitar.wav'
#spectrum_num = 13

sound_array, fs = read_wav(filename, mmap=False)
sound_array = sound_array[0: fs*8]

n, sound_array = add_zeros(sound_array, degree=12)  # where n = 2^degree

duration = len(sound_array) / fs  # -> duration of the whole audio.
dt = 1 / fs
time_vector = arange(0, duration, dt)
clean = deepcopy(sound_array)
sound_array = sound_array + 30000*sin(2*pi*3000*time_vector) + 22000*sin(2*pi*4000*time_vector) + 25000*sin(2*pi*2000*time_vector)+ 25000*sin(2*pi*1000*time_vector)
degree = 10
fft_points = 2 ** degree
#L = np.arange(1, np.floor(fft_points/2), dtype='int')
scaledd = int16(sound_array / max(abs(sound_array)) * 32767)
write("D:\\Audio_files\\noisy.wav", fs, scaledd)


#doing FFT
PSD = fft_fft(sound_array=sound_array, time_array=time_vector, fft_points=fft_points)

freq = (1/(dt*fft_points)) * arange(fft_points)
d_freq = freq[-1] / len(freq)
"""
y_filt_1 = band_stop(lowcut=900, highcut=1100, order=4, fs=fs, signal_array=sound_array)
y_filt_2 = band_stop(lowcut=1900, highcut=2100, order=4, fs=fs, signal_array=y_filt_1)
y_filt_3 = band_stop(lowcut=2900, highcut=3100, order=4, fs=fs, signal_array=y_filt_2)
y_filt_4 = band_stop(lowcut=3900, highcut=4100, order=4, fs=fs, signal_array=y_filt_3)
"""

y_filt_4 = band_stop_with_list(lowcut=[900,1900,2900,3900], highcut=[1100,2100,3100,4100], order=4, fs=fs, signal_array=sound_array)

PSD2 = fft_fft(sound_array=y_filt_4, time_array=time_vector, fft_points=fft_points)


#scaled2 = np.int16(abs(y_filt2) / np.max(np.abs(y_filt2)) * 32767)
write("D:\\Audio_files\\filtered_audio.wav", fs, y_filt_4.astype(int16))


plt.figure(1)
plt.subplot(411)
plt.plot(time_vector, clean, label='clean')
plt.legend()

plt.subplot(412)
for j in range(len(PSD)):
    plt.plot(freq, PSD[j])
    plt.xlim(0, max(freq)/2)

plt.subplot(413)
plt.plot(time_vector, sound_array, label='noisy')
plt.legend()
plt.subplot(414)
plt.plot(time_vector, y_filt_4, label='filtered')
plt.ylim(-27000, 34000)
#plt.subplot(414)
#plt.plot(time_vector, y_filt2)
plt.legend()
plt.figure(2)
plt.subplot(211)
for j in range(len(PSD2)):
    plt.plot(freq, PSD2[j])
    plt.xlim(0, max(freq)/2)
plt.subplot(212)
for j in range(len(PSD)):
    plt.plot(freq, PSD[j])
    plt.xlim(0, max(freq)/2)

plt.savefig("clean_psd")
plt.show()

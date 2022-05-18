from functions import *
from numpy import arange, int16
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

filename = 'D:\\Audio_files\\simulation.wav'
#spectrum_num = 13

sound_array, fs = read_wav(filename, mmap=False)
#sound_array = sound_array[0: fs*8]

n, sound_array = add_zeros(sound_array, degree=12)  # where n = 2^degree

duration = len(sound_array) / fs  # -> duration of the whole audio.
dt = 1 / fs
time_vector = arange(0, duration, dt)
degree = 10
fft_points = 2 ** degree
#L = np.arange(1, np.floor(fft_points/2), dtype='int')
scaledd = int16(sound_array / np.max(np.abs(sound_array)) * 32767)
write("D:\\Audio_files\\noisy.wav", fs, scaledd)

#doing FFT
PSD = fft_fft(sound_array=sound_array, time_array=time_vector, fft_points=fft_points)

freq = (1 / (dt*fft_points)) * np.arange(fft_points)
d_freq = freq[-1] / len(freq)

#y_filt_4 = band_pass(lowcut=10, highcut=8000, order=4, fs=fs, signal_array=sound_array)

y_filt_4 = band_stop_with_list(lowcut=[10, 6000], highcut=[3000, 7000], order=4, fs=fs, signal_array=sound_array)

PSD2 = fft_fft(sound_array=y_filt_4, time_array=time_vector, fft_points=fft_points)

res = summing_specs(PSD)


y_filt_4 = y_filt_4 * 100000

#scaled2 = np.int16(abs(y_filt2) / np.max(np.abs(y_filt2)) * 32767)
write("D:\\Audio_files\\filtered_audio.wav", fs, y_filt_4.astype(np.int16))

plt.figure(1)
plt.subplot(311)
plt.plot(time_vector, sound_array, label='noisy')
plt.legend()
plt.subplot(312)
for j in range(len(PSD2)):
    plt.plot(freq, PSD2[j])
    plt.xlim(0, max(freq)/2)
plt.subplot(313)
plt.plot(time_vector, y_filt_4, label='filtered')
plt.ylim(-27000, 34000)
plt.legend()
plt.figure(2)
plt.plot(freq, res, label='clean')
plt.xlim(0, max(freq)/2)
plt.legend()
plt.savefig("clean_psd")
plt.show()
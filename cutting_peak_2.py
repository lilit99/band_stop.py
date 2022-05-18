from functions import read_wav, add_zeros, find_local_peaks,sort_index, find_peak_widths, find_points, find_exact_point, conflict_solver_1
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io.wavfile import write


filename = 'D:\\Audio_files\\Microphone_test.wav'
spectrum_num = 34
sound_array, fs = read_wav(filename, mmap=False)
#sound_array = sound_array[0: fs*8]
n, sound_array = add_zeros(sound_array, degree=12)  # where n = 2^degree

duration = len(sound_array) / fs  # -> duration of the whole audio.
dt = 1 / fs
time_vector = np.arange(0, duration, dt)

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
    PSD_fft = np.fft.fft(sliced_sound_array[j])
    PSD.append(abs(PSD_fft))

freq = (1/(dt*fft_points)) * np.arange(fft_points)
d_freq = freq[-1] / len(freq)
#freq = np.fft.fftfreq(fft_points, d=1/fs)


peak_index, _ = find_local_peaks(PSD[spectrum_num])



width, _, left_ips, right_ips = find_peak_widths(PSD[spectrum_num], peak_index)
width = d_freq * width
left_ips = d_freq * left_ips
right_ips = d_freq * right_ips

height = np.interp(freq[peak_index], freq, fp=PSD[spectrum_num])
height = height/2
freq = list(freq)
right = []
left = []
for i in range(len(peak_index)):
    right.append(right_ips[i] - freq[peak_index[i]])
    left.append(freq[peak_index[i]] - left_ips[i])

filt_right = []
filt_left = []
filt_right_index = []
filt_left_index = []
for i in range(len(peak_index)):
    r_list = []
    l_list = []
    numr = freq[peak_index[i]] + right[i]*3
    numl = freq[peak_index[i]] - left[i]*3
    for j in range(len(freq)):
        r_list.append(abs(numr - freq[j]))
        l_list.append(abs(numl - freq[j]))
    filt_right.append(min(r_list))
    filt_right_index.append(r_list.index(min(r_list)))
    filt_left_index.append(l_list.index(min(l_list)))
    filt_left.append(min(l_list))

max_height = sort_index(PSD[spectrum_num][peak_index])

PSD_filtered = deepcopy(PSD)
for i in max_height[0:7]:
    PSD_filtered[spectrum_num][filt_left_index[i]:filt_right_index[i]] = PSD_filtered[spectrum_num][filt_left_index[i]+1]/2 + PSD_filtered[spectrum_num][filt_left_index[i]-1]/2


plt.figure(1)
plt.plot(time_vector, sound_array)
plt.figure(2)
plt.subplot(211)
plt.plot(freq, PSD[spectrum_num])
plt.xlim(0, 2000)
plt.ylim(0, np.max(PSD[spectrum_num]) + np.max(PSD[spectrum_num])/6)
plt.subplot(212)
plt.plot(freq,PSD_filtered[spectrum_num])
plt.xlim(0, 2000)
plt.ylim(0, np.max(PSD[spectrum_num]) + np.max(PSD[spectrum_num])/6)
plt.savefig('spectrum')
plt.show()

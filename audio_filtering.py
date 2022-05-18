from functions import read_wav, add_zeros, find_local_peaks,sort_index, find_peak_widths, find_points, find_exact_point
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io.wavfile import write

sound_array, fs = read_wav('D:\\Audio_files\\Microphone_test.wav', mmap=False)
sound_array = sound_array[0: fs*8]
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
    PSD.append(np.fft.fft(sliced_sound_array[j]))

freq = (1/(dt*fft_points)) * np.arange(fft_points)

filtered_powers = deepcopy(PSD)

d_freq = freq[-1] / len(freq)

#for i in range(len(PSD)):
    #for j in range(len(PSD[0])):
      #  PSD[i][j] = abs(PSD[i][j])

peak_index_list = []
max_indexis = []
filtered_powers = deepcopy(PSD)
frequenciesFound = list(freq)
for j in range(len(PSD)):
    max_height = sort_index(filtered_powers[j])
    max_indexis.append(max_height)
for j in range(len(PSD)):
    peak_index, _ = find_local_peaks(PSD[j])
    peak_index_list.append(peak_index)

width_list = []
height_list = []
x_right_list = []
x_left_list = []

for i in range(len(PSD)):
    width, _, left_ips, right_ips = find_peak_widths(PSD[i], peak_index_list[i] )  # -> finding local peaks's widths

    height = np.interp(freq[peak_index_list[i]], frequenciesFound, fp=PSD[i])
    height = height/2
    height_list.append(height)
    width = d_freq * width
    left_ips = d_freq * left_ips
    right_ips = d_freq * right_ips
    width_list.append(width)
    right_part = right_ips - freq[peak_index_list[i]]
    left_part = freq[peak_index_list[i]] - left_ips

# diction = {frequenciesFound[peak_index][i]: width[i] for i in range(len(width))}  # -> creating a dictionary
# keeping peak index, width and height in one list

    peak_left_x1, peak_left_x2, peak_left_y1, peak_left_y2, peak_right_x1, peak_right_x2, peak_right_y1, peak_right_y2 = find_points(peak_index=peak_index_list[i],
                                                    power=PSD[i], height=PSD[i][peak_index_list[i]], frequencies=frequenciesFound)

    x_right, x_left = find_exact_point(peak_left_x1, peak_left_x2, peak_left_y1, peak_left_y2, peak_right_x1, peak_right_x2,peak_right_y1,peak_right_y2,height)
    x_right_list.append(peak_right_x1)
    x_left_list.append(peak_left_x1)

width_coords = []
for i in range(len(peak_index_list)):
    for j in range(len(peak_index_list[i])):
        width_coords.append(x_left_list[i][j])
       # width_coords.append(x_left_list[i][j])
        width_coords.append(x_right_list[i][j])

for l in range(len(PSD)):
    for k in range(len(PSD)):
        if abs(PSD[l][k]) > abs(PSD[l][max_indexis[l][6]]):
            PSD[l][k] = 0
        #PSD[l][frequenciesFound.index(x_left_list[l][k]) : frequenciesFound.index(x_right_list[l][k])] = 0
        #if abs(PSD[l][k]) > abs(PSD[l][max_indexis[l][2]]):
            #PSD[l][frequenciesFound.index(x_left_list[l][list(peak_index_list[l]).index(k)]) : frequenciesFound.index(x_right_list[l][list(peak_index_list[l]).index(k)])] = 0
            #PSD[l][k] = 0

filt_list = []
for j in range(len(PSD)):
    ffilt = np.fft.ifft(PSD[j])
    filt_list.append(ffilt)

fil = []
for j in range(len(filt_list)):
    for q in filt_list[j]:
        fil.append(q)
fil = np.array(fil)

scaled = np.int16(fil / np.max(np.abs(fil)) * 32767)
write("D:\\Audio_files\\filtered_audio.wav", fs, scaled)

plt.figure(1)
plt.subplot(411)
plt.plot(time_vector, sound_array)
plt.xlabel('Time(sc)')
plt.ylabel("Sound")
plt.legend()

plt.subplot(412)
for j in range(len(filtered_powers)):
    plt.plot(freq, abs(filtered_powers[j]))
    plt.xlim(0, max(freq)/2)
plt.xlabel("frequency")
plt.ylabel("PSD")
plt.legend()
plt.subplot(414)
p = 0
while p < len(sliced_time_array):
    plt.plot(sliced_time_array[p:p+fft_points], filt_list[p:p+fft_points], color='c')
    p += fft_points
plt.xlabel("Time (sec)")
plt.ylabel("Sound")
#plt.legend()
plt.subplot(413)
for j in range(len(PSD)):
    plt.plot(freq, abs(PSD[j]))
    plt.xlim(0, max(freq)/2)
    plt.xlim(0, max(freq/2))
    plt.ylim(0, 40000)
plt.show()
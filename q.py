# Doing FFT and IFFT (without any filter)
"""
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
inverse_peak_list = []
max_indexis = []
filtered_powers = deepcopy(PSD)
frequenciesFound = list(freq)

for j in range(len(PSD)):
    peak_index, _ = find_local_peaks(PSD[j])
    inverse, _ = find_local_peaks(-1 * PSD[j])
    inverse = list(inverse)
    peak_index = list(peak_index)
    peak_index_list.append(peak_index)
    inverse_peak_list.append(inverse)


for i in range(len(peak_index_list)):
    max_height = sort_index(PSD[i][peak_index_list[i]])
    max_indexis.append(max_height)



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
write("D:\\Audio_files\\filtered_Mic_test", fs, scaled)

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
plt.legend()
plt.subplot(413)
for j in range(len(PSD)):
    plt.plot(freq, abs(PSD[j]))
    plt.xlim(0, max(freq)/2)
    plt.xlim(0, max(freq/2))
    plt.ylim(0, 40000)
plt.show()
"""

from functions import read_wav, add_zeros, find_local_peaks,sort_index, find_peak_widths, find_points, find_exact_point, conflict_solver_1
import numpy as np
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
time_vector = np.arange(0, duration, dt)
sound_array = sound_array + 30000*np.sin(2*np.pi*3000*time_vector) + 20000*np.sin(2*np.pi*4000*time_vector)+150000*np.sin(2*np.pi*2000*time_vector)
degree = 10
fft_points = 2 ** degree
#L = np.arange(1, np.floor(fft_points/2), dtype='int')
scaledd = np.int16(sound_array / np.max(np.abs(sound_array)) * 32767)
write("D:\\Audio_files\\esim.wav", fs, scaledd)
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

peak_index_list = []
width_list = []
right_list = []
left_list = []

filt_right_index_list = []
filt_left_index_list = []
max_height_list =[]
PSD_filtered = deepcopy(PSD)
for i in range(len(PSD)):
    peak_index, _ = find_local_peaks(PSD[i])
    peak_index_list.append(peak_index)

    width, _, left_ips, right_ips = find_peak_widths(PSD[i], peak_index)
    width = d_freq * width
    left_ips = d_freq * left_ips
    right_ips = d_freq * right_ips

    width_list.append(width)
    hhh = PSD[i][peak_index]
    #height = np.interp(freq[0], freq, fp=PSD[i])
    #height = height/2
    freq = list(freq)
    right = []
    left = []
    for j in range(len(peak_index)):
        right.append(right_ips[j] - freq[peak_index[j]])
        left.append(freq[peak_index[j]] - left_ips[j])
    right_list.append(right)
    left_list.append(left)

    filt_right = []
    filt_left = []
    filt_right_index = []
    filt_left_index = []
    for k in range(len(peak_index)):
        r_list = []
        l_list = []
        numr = freq[peak_index[k]] + right[k] * 3
        numl = freq[peak_index[k]] - left[k] * 3
        for e in range(len(freq)):
            r_list.append(abs(numr - freq[e]))
            l_list.append(abs(numl - freq[e]))

        filt_right.append(min(r_list))
        filt_right_index.append(r_list.index(min(r_list)))
        filt_left_index.append(l_list.index(min(l_list)))
        filt_left.append(min(l_list))
    filt_left_index_list.append(filt_left_index)
    filt_right_index_list.append(filt_right_index)

    max_height = sort_index(PSD[i][peak_index])
    for t in max_height[0:4]:
        PSD_filtered[i][filt_left_index[t] : filt_right_index[t]] = PSD_filtered[i][filt_left_index[t]+1]/2 + PSD_filtered[i][filt_left_index[t]-1]/2

filt_list = []
for j in range(len(PSD_filtered)):
    ffilt = np.fft.ifft(PSD_filtered[j])
    filt_list.append(ffilt)

fil = []
for j in range(len(filt_list)):
    for q in filt_list[j]:
        fil.append(q)
fil = np.array(fil)
scaled = np.int16(abs(fil) / np.max(np.abs(fil)) * 32767)
write("D:\\Audio_files\\filt_microphone.wav", fs, scaled)

plt.figure(1)
plt.plot(time_vector, sound_array)
plt.figure(2)
plt.subplot(211)
for j in range(len(PSD)):
    plt.plot(freq, PSD[j])
    #plt.xlim(0, max(freq)/2)

plt.xlim(0, max(freq)/2)
#plt.ylim(0, np.max(PSD) + np.max(PSD)/6)
plt.subplot(212)

#plt.ylim(0, np.max(PSD[spectrum_num]) + np.max(PSD[spectrum_num])/6)
for j in range(len(PSD_filtered)):
    plt.plot(freq, abs(PSD_filtered[j]))
plt.xlim(0, max(freq)/2)
#plt.ylim(0, 40000)

plt.savefig('spectrum')
plt.show()
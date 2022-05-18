from functions import find_local_peaks, read_wav, find_peak_widths, add_zeros, find_points, find_exact_point,sort_index
import matplotlib.pyplot as plt
import numpy as np
import sys
from copy import deepcopy

np.set_printoptions(threshold=sys.maxsize)

# reading wav file
sound_array, fs = read_wav('D:\\Audio_files\\examples_samples_guitar.wav', mmap=False)
# fs = 16.000
# len(sound_array) = 166440:

n, sound_array = add_zeros(sound_array, degree=12)  # where n = 2^degree
spectrum_number = 13 # --> which spectrum to see
duration = len(sound_array) / fs  # -> duration of the whole audio.
time_vector = np.arange(0, duration, 1 / fs)

# fft points
freq = np.fft.fftfreq(n, d=1.0 / fs)  # -> Return the DFT sample frequencies.
# slicing time

dtime = int(len(sound_array) / n)
time = np.arange(0, len(sound_array), dtime)

# .......................................................................................................................
# plotting signal
plt.figure(1)
plt.subplot(211)
#plt.plot(time_vector, sound_array)  # --> signal
#plt.xlim([0, np.max(time_vector)])
#plt.xlabel('Time')
#plt.ylabel('Amplitude')

# plotting spectrogram
#plt.subplot(212)
powerSpectrum, frequenciesFound, t, imageAxis = plt.specgram(sound_array, Fs=fs, mode='magnitude', noverlap=0, NFFT=n,
                                                             xextent=(0, np.max(time_vector)))
d_freq = frequenciesFound[-1] / len(frequenciesFound)
plt.xlabel('Time')
plt.ylabel('Frequency')
#cb = plt.colorbar(orientation="horizontal")

# .......................................................................................................................

powerS = np.transpose(powerSpectrum)

peak_index, properties = find_local_peaks(powerS[spectrum_number])  # -> finding local peaks

width, _, left_ips, right_ips = find_peak_widths(powerS[spectrum_number], peak_index )  # -> finding local peaks's widths
height = np.interp(frequenciesFound[peak_index], frequenciesFound, fp=powerS[spectrum_number])
height = height/2

width = d_freq * width
left_ips = d_freq * left_ips
right_ips = d_freq * right_ips


right_part = right_ips - frequenciesFound[peak_index]
left_part = frequenciesFound[peak_index] - left_ips

# diction = {frequenciesFound[peak_index][i]: width[i] for i in range(len(width))}  # -> creating a dictionary
# keeping peak index, width and height in one list

peak_left_x1, peak_left_x2, peak_left_y1, peak_left_y2, peak_right_x1, peak_right_x2, peak_right_y1, peak_right_y2 = find_points(peak_index=peak_index,
                                                    power=powerS[spectrum_number], height=height, frequencies=frequenciesFound)

x_right, x_left = find_exact_point(peak_left_x1, peak_left_x2, peak_left_y1, peak_left_y2, peak_right_x1, peak_right_x2,peak_right_y1,peak_right_y2,height)
print(peak_right_x1)
print(peak_left_x1)
width_coords = []
newheight = []

for j in range(len(peak_index)):
    width_coords.append(x_left[j])
    newheight.append(0)
    width_coords.append(x_left[j])
    width_coords.append(x_right[j])
    newheight.append(height[j])
    newheight.append(height[j])
    width_coords.append(x_right[j])
    newheight.append(0)


peak_width_height = np.zeros((len(peak_index), 3))
for j in range(0, len(peak_index)):
    peak_width_height[j][0] = peak_index[j]
    peak_width_height[j][1] = x_right[j] - x_left[j]
    peak_width_height[j][2] = height[j]
#print(peak_width_height)

heights = 2 * height
max_indexis = sort_index(heights)[:5]
filtered_powers = deepcopy(powerS)
frequenciesFound = list(frequenciesFound)



for i in max_indexis:
    filtered_powers[spectrum_number][frequenciesFound.index(peak_left_x1[i]) : frequenciesFound.index(peak_right_x1[i])] = height[i+3]

# .....................................................................................
#  for key, value in diction.items():
#  print(key, ' : ', value)
#  print(len(frequenciesFound))
#  plot PSD

plt.subplot(212)

plt.plot(frequenciesFound, powerS[spectrum_number], color='c', label='PSD')
#plt.plot(width_coords, newheight, color='black')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.legend()

plt.xlim(0, (frequenciesFound[-1]) / 4)
plt.ylim(0, np.max(powerS[spectrum_number]) + 100)

plt.figure(2)
plt.plot(frequenciesFound, filtered_powers[spectrum_number],color='c')
print(peak_left_x1)
print(peak_right_x1)
plt.xlim(0, (frequenciesFound[-1]) / 4)
plt.ylim(0, np.max(powerS[spectrum_number]) + 100)
plt.figure(3)
plt.plot(frequenciesFound, powerS[spectrum_number], color='c', label='PSD')
plt.xlim(0, (frequenciesFound[-1]) / 4)
plt.ylim(0, np.max(powerS[spectrum_number]) + 100)
#plt.savefig('psd_plot_1.png')

plt.show()
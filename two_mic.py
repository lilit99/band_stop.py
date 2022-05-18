from functions import *
from functions import *
from numpy import arange, int16, array
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

filename1 = 'D:\\Audio_files\\simulation_results\\Microphone1.wav'
filename2 = 'D:\\Audio_files\\simulation_results\\Microphone2.wav'

sound_array1, fs1 = read_wav(filename1, mmap=False)
sound_array2, fs2 = read_wav(filename2, mmap=False)

duration = len(sound_array1) / fs1  # -> duration of the whole audio.
dt = 1 / fs1
time_vector = arange(0, duration, dt)

new_sound = []
for element in range(len(sound_array2)):
    new_sound.append(sound_array2[element] - sound_array1[element])

new_sound = array(new_sound)
scal = int16(new_sound / max(abs(new_sound)) * 32767)
write("D:\\Audio_files\\filt_2_mic.wav", fs1, scal)

plt.figure(1)
plt.plot(time_vector, sound_array1)
plt.plot(time_vector, sound_array2)

plt.figure(2)
plt.plot(time_vector, new_sound)
plt.show()
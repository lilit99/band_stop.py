from numpy import arange, int16, max, transpose, pad, zeros, amax, fft, abs, add
import matplotlib.pyplot as plt
import scipy.io as sp
from scipy.io.wavfile import write
from scipy.signal import find_peaks, peak_widths, peak_prominences
import math
import scipy.signal
import pyaudio
import wave
import time
from scipy import signal

def read_wav(filename="", mmap=True):
    """
    :param filename: string filename
    :param mmap: bool, two-channel or one-channel
    :return: sound_array, fs
    """
    fs, audio = sp.wavfile.read(filename)
    sound_array = []
    if mmap:  # if form is true, the wav file is interleaved stereo file. takes 1st channel
        for i in range(0, len(audio)):
            sound_array.append(audio[i][1])
    else:
        sound_array = audio
    return sound_array, fs

def resamplingAudio(audio, fs, new_fs):
    # taking the nearest integer duration of the signal
    n = math.ceil(len(audio) / fs)
    # Counting the number of added points to the initial signal
    add_to_len = n * fs - len(audio)
    if add_to_len != 0:
        # Adding additional points
        addition = [0] * add_to_len
        audio += addition
    # Number of samples for resampled audio waveform
    new_audio_samples = int(new_fs * len(audio) / fs)

    new_audio = scipy.signal.resample(audio, new_audio_samples)
    return new_audio

#...........................................................................................................

play_pause = []
stop = []
def player(filename, new_fs: int, n: int):
    """
    player function modifies your audio's sample frequency, divides an audio into n parts, and for each part
    checks if play, pause and stop buttons are pressed or not.
    :param filename: File name which fs you want to modify
    :param new_fs: Your desired sample frequency
    :param n: Divide an audio into n parts
    :return:
    """
    sound_array, fs = read_wav(filename, mmap=False)
    audio = resamplingAudio(list(sound_array), fs, new_fs)
    write("audio.wav", new_fs, audio.astype(int16))

    file_name = wave.open("audio.wav", 'rb')

    length_n = int(len(audio) / n)  # length of each part of a divided audio
    duration = int(len(audio) / new_fs)  # duration of an audio
    time_array = arange(0, duration, duration / len(audio))

    # doing fft
    power_spectrum, frequencies_found, t, image_axis = plt.specgram(audio, Fs=new_fs,
                                                                    mode='magnitude',
                                                                    noverlap=0,
                                                                    NFFT=1024,
                                                                    xextent=(0, max(time_array)))
    powers = transpose(power_spectrum)

    global play_pause
    global stop
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(format=py_audio.get_format_from_width(file_name.getsampwidth()),
                           channels=file_name.getnchannels(),
                           rate=file_name.getframerate(),
                           output=True)

    i = 0
    while i < len(audio) - length_n:
        if stop == 1:
            i = 0
        while play_pause == 0:
            time.sleep(1)
        file_name.setpos(i)
        frames = file_name.readframes(length_n)
        stream.write(frames)
        minute_second = time_array[i]
        power = powers[i]
        frequency = frequencies_found
        # return minute_second
        i += length_n

    stream.close()
    py_audio.terminate()
    file_name.close()

def add_zeros(array: [], degree: int):
    """
    :param array: array in which we are adding zeros
    :param degree: what degree of two do we want the length of the array to be multiple of
    :return: n, array
    """
    n = 2 ** degree
    zeros = (int(len(array) / (2 ** degree)) + 1) * (2 ** degree) - len(array)  # -> number of zeros
    array = pad(array, (0, zeros), 'constant')
    return n, array

# ......................................................................................................

def find_local_peaks(power: []):
    """
    :param power: list of powers in each frequency
    :return: peak_index, properties
    """
    peak_index, properties = find_peaks(power)
    return peak_index, properties

# ......................................................................................................

def find_peak_widths(power: [], index: int):
    """
    local peaks's widths
    :param power:
    :param index: peaks indexes
    :return: width, heights, left_ips, right_ips
    """
    width, _, left_ips, right_ips = peak_widths(power, index, rel_height = 0.5)
    return width, _, left_ips, right_ips

#.....................................................................................................

def find_coords(freq: [], index: int, x: [], y: [], left_ips: [], right_ips: [], heights: []):
    """

    :param freq:
    :param index:
    :param x:
    :param y:
    :param left_ips:
    :param right_ips:
    :param heights:
    :return:
    """
    for i in range(len(index)):
        j = 0
        while (freq[index][i - 1] + right_ips[i - 1]) + j < freq[index][i] - left_ips[i]:
            x.append(freq[index][i - 1] + right_ips[i - 1] + j)
            y.append(0)
            j += 1
        x.append(freq[index][i] - left_ips[i])
        y.append(heights[i])
        x.append(freq[index][i] + right_ips[i])
        y.append(heights[i])
    return x, y

#...................................................................................................
def find_points(peak_index:[], power:[], height:[], frequencies:[]):
    peak_right_x1 = []
    peak_right_x2 = []
    peak_left_x1 = []
    peak_left_x2 = []
    peak_left_y1 = []
    peak_left_y2 = []
    peak_right_y1 = []
    peak_right_y2 = []
    for i in range(len(peak_index)):
        forward = peak_index[i]
        backward = peak_index[i]
        while power[forward] > height[i]:
            forward += 1
            if forward > len(frequencies) - 2:
                break
        peak_right_x1.append(frequencies[forward - 1])
        peak_right_x2.append(frequencies[forward])
        peak_right_y1.append(power[forward - 1])
        peak_right_y2.append(power[forward])
        while power[backward] > height[i]:
            backward -= 1
            if backward < 0:
                break
        peak_left_x1.append(frequencies[backward + 1])
        peak_left_x2.append(frequencies[backward])
        peak_left_y1.append(power[backward + 1])
        peak_left_y2.append(power[backward])
    return peak_left_x1, peak_left_x2, peak_left_y1,  peak_left_y2,peak_right_x1, peak_right_x2, peak_right_y1,peak_right_y2
#...........................................................................................................


def find_exact_point(peak_left_x1,peak_left_x2,peak_left_y1,peak_left_y2,peak_right_x1, peak_right_x2,peak_right_y1,peak_right_y2,height):
    a = zeros(len(peak_left_x1))
    b = zeros(len(peak_left_x1))

    x_left = zeros(len(peak_left_x1))
    x_right = zeros(len(peak_right_x1))

    for w in range(len(peak_left_x1)):
        a[w] = (peak_left_y2[w] - peak_left_y1[w]) / (peak_left_x2[w] - peak_left_x1[w])
        b[w] = peak_left_y1[w] - (a[w] * peak_left_x1[w])

        x_left[w] = (height[w] - b[w]) / a[w]

    c = zeros(len(peak_left_x1))
    d = zeros(len(peak_left_x1))

    for i in range(len(peak_right_x1)):
        c[i] = (peak_right_y2[i] - peak_right_y1[i]) / (peak_right_x2[i] - peak_right_x1[i])
        d[i] = peak_right_y1[i] - c[i] * peak_right_x1[i]

        x_right[i] = (height[i] - d[i]) / c[i]

    return x_right, x_left

#..................................................................................................................
def combine_peaks(peak_index:[],x_right, x_left,frequenciesFound,width):
    combo = []
    width_combo = []
    for i in range(1, len(peak_index) - 2):
        h = i + 1
        m = i - 1
        count = 1
        count2 = 1
        while x_right[i] > x_left[h]:
            count += 1
            h += 1
        while x_left[i] < x_right[m]:
            m -= 1
            count2 += 1
            if m < 0:
                break
        combo.append(frequenciesFound[peak_index[(i-count2):(i + count)]])
        width_combo.append(width[(i-count2):(i + count)])

    combined_list = zeros((len(peak_index), 2), dtype=object)

    for i in range(len(combo)):
        combined_list[i][0] = combo[i]
        combined_list[i][1] = width_combo[i]

    #for i in combined_list:
      #  for j in i:
          #  print(j, end="")
      #  print()

    return combined_list,combo

#..................................................................................................................

def sort_index(lst, rev=True):
    index = range(len(lst))
    max_indexis = sorted(index, reverse=rev, key=lambda i: lst[i])
    return max_indexis
#..................................................................................................................
def conflict_solver_1(peak_index, x_right, x_left, height):

    new_x_right = []
    new_x_left = []
    new_height = []
    new_peak_index = []
    i = 0
    while i < len(peak_index) - 23:
        h = i + 1
        count = 1
        while x_right[i] > x_left[h]:
            h += 1
            count += 1
        keep = amax(height[i: i + count])
        height = list(height)
        keep_index = height.index(keep)
        new_x_right.append(x_right[keep_index])
        new_x_left.append(x_left[keep_index])
        new_height.append(height[keep_index])
        new_peak_index.append(peak_index[keep_index])
        i += count
    return new_x_right, new_x_left, new_height, new_peak_index

#........................................................................


def band_stop(lowcut, highcut, order, fs, signal_array):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandstop')
    filtered_signal = signal.filtfilt(b, a, signal_array)
    return filtered_signal

#........................................................................


def band_stop_with_list(lowcut:[], highcut:[], order, fs, signal_array):
    nyq = 0.5 * fs
    low = zeros(len(lowcut))
    high = zeros(len(highcut))
    for i in range(len(lowcut)):
        low = lowcut[i] / nyq
        high = highcut[i] / nyq

        b, a = signal.butter(order, [low, high], btype='bandstop')
        filtered_signal = signal.filtfilt(b, a, signal_array)
        signal_array = filtered_signal
    return filtered_signal

def band_pass(lowcut, highcut, order, fs, signal_array):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filtered_signal = signal.filtfilt(b, a, signal_array)
    return filtered_signal

def fft_fft(sound_array, time_array, fft_points):
    sliced_sound_array = []
    sliced_time_array = []
    i = 0
    while i < len(sound_array):
        sliced_sound_array.append(sound_array[i: i + fft_points])
        sliced_time_array.append(time_array[i: i + fft_points])
        i += fft_points
    PSD = []
    for j in range(len(sliced_sound_array)):
        PSD_fft = abs(fft.fft(sliced_sound_array[j]))
        PSD.append(PSD_fft)
    return PSD

#.......................................................................................

def summing_specs(PSD:[]):
    res = zeros(len(PSD[0]))
    for spec in PSD:
        res = add(res, spec)
    return res











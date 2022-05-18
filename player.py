import numpy as np
from functions import read_wav, resamplingAudio
import threading
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import pyaudio
import wave
import time

filename = 'D:\\Audio_files\\examples_samples_guitar.wav'


def player(filename=filename, n=34, new_fs=32000):
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
    write("audio.wav", new_fs, audio.astype(np.int16))

    file_name = wave.open("audio.wav", 'rb')
    length_n = int(len(audio) / n)  # length of each part of a divided audio
    duration = int(len(audio) / new_fs)  # duration of an audio
    time_array = np.arange(0, duration, duration / len(audio))

    #  doing fft
    power_spectrum, frequencies_found, _, _ = plt.specgram(audio, Fs=new_fs,
                                                           mode='magnitude',
                                                           noverlap=0,
                                                           NFFT=1024,
                                                           xextent=(0, np.max(time_array)))
    powers = np.transpose(power_spectrum)
    global play_pause
    global stop
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(format=py_audio.get_format_from_width(file_name.getsampwidth()),
                           channels=file_name.getnchannels(),
                           rate=file_name.getframerate(),
                           output=True)

    i = 0
    while i < len(audio) - length_n:
        time.sleep(0.000001)
        if stop == 1:
            i = 0
        while play_pause == 0:
            time.sleep(1)

        file_name.setpos(i)
        frames = file_name.readframes(length_n)
        stream.write(frames)
        i += length_n

    stream.close()
    py_audio.terminate()
    file_name.close()


if __name__ == "__main__":
    play_pause = 1
    stop = 0
    t = threading.Thread(target=player)
    t.start()
    time.sleep(3)
    play_pause = 0
    time.sleep(2)
    play_pause = 1
    time.sleep(1)
    stop = 1
    time.sleep(3)
    stop = 0

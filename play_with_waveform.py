from functions import read_wav, resamplingAudio
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
from multiprocessing import Process
import wave
from scipy.io.wavfile import write

filename = 'D:\\Audio_files\\examples_samples_guitar.wav'
sound_array, fs = read_wav(filename, mmap=False)
new_fs = 32000
n = 25
audio = resamplingAudio(list(sound_array), fs, new_fs)
write("audio.wav", new_fs, audio.astype(np.int16))

file_name = wave.open("audio.wav", 'rb')
length_n = int(len(audio) / n)  # length of each part of a divided audio
duration = int(len(audio) / new_fs)  # duration of an audio
time_array = np.arange(0, duration, duration / len(audio))

#  doing fft
power_spectrum, frequencies_found, t, image_axis = plt.specgram(audio, Fs=new_fs,
                                                                mode='magnitude',
                                                                noverlap=0,
                                                                NFFT=1024,
                                                                xextent=(0, np.max(time_array)))
powers = np.transpose(power_spectrum)
play_pause = 1
stop = 0

def player(filename=filename, n=n, fs=new_fs):

    """
    player function modifies your audio's sample frequency, divides an audio into n parts, and for each part
    checks if play, pause and stop buttons are pressed or not.
    :param filename: File name which fs you want to modify
    :param new_fs: Your desired sample frequency
    :param n: Divide an audio into n parts
    :return:
    """
    global play_pause
    global stop
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(format=py_audio.get_format_from_width(file_name.getsampwidth()),
                           channels=file_name.getnchannels(),
                           rate=file_name.getframerate(),
                           output=True)

    i = 0
    while i < len(audio) - length_n:
        #time.sleep(0.000001)
        #if stop == 1:
        #i = 0
        # while play_pause == 0:
        #time.sleep(1)

        file_name.setpos(i)
        frames = file_name.readframes(length_n)
        stream.write(frames)
        i += length_n

    stream.close()
    py_audio.terminate()
    file_name.close()

def showing_audiotrack():
    previousTime = time.time()
    plt.ion()
    spentTime = 0
    period = 0.5

    #  plotting audio
    for i in range(len(audio)):
        if i % 16000 == 0:
            spentTime += 0.5
        # update plot every period minute
        if spentTime == period:
            # Clear the previous plot
            plt.clf()
            # Plot the audio data
            plt.plot(time_array, audio)
            # line
            plt.axvline(x=i / new_fs, color='r')
            plt.xlabel("Time (s)")
            plt.ylabel("Audio")
            plt.show()
            plt.pause(period-(time.time()-previousTime))
            previousTime = time.time()
            spentTime = 0

if __name__=="__main__":
    p1 = Process(target=player, args=())
    p1.start()
    p2 = Process(target=showing_audiotrack, args=())
    p2.start()
    p1.join()
    p2.join()

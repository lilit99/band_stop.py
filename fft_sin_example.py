import numpy as np
import matplotlib.pyplot as plt
import copy

dt = 0.001
t = np.arange(0,2,dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
f_clean = copy.deepcopy(f)
f = f + 2.5*np.random.randn(len(t))
#compute FFT

n = len(t)
fhat = np.fft.fft(f, n)
PSD = fhat * np.conj(fhat) / n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype='int')


indices = PSD > 100
PSDclean = PSD * indices
ffhat = indices * fhat

ffilt = np.fft.ifft(ffhat)

plt.figure(1)
plt.subplot(411)
plt.plot(t, f, color='c', label="Noisy")
plt.plot(t, f_clean, color='k', label='clean')
plt.legend()
plt.subplot(412)
plt.plot(freq, PSD, color='c', label="noisy")
plt.xlim(0, max(freq)/2)
plt.ylim(-27, 426)
plt.legend()
plt.subplot(413)
plt.xlim(0, max(freq)/2)
plt.ylim(-27, 426)
plt.plot(freq, PSDclean)
plt.subplot(414)
plt.plot(t, ffilt, color='k', label='clean')
plt.ylim(-10, 10)
plt.legend()
plt.show()


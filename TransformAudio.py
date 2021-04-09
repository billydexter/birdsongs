import sys
import librosa
import matplotlib.pyplot as plt
from librosa import display
from scipy.fft import fft
import numpy as np
import os

def fft_plot(audio, sampling_rate):
    n = len(audio)
    T = 1 / sampling_rate
    yf = fft(audio)
    xf = np.linspace(0, int(1/(2*T)), int(n/2))
    fig, ax = plt.subplots()
    repYf = 2.0/n * np.abs(yf[:n//2])
    ax.plot(xf, repYf)
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    return plt.show()


def spectrogram(samples, sample_rate, stride_ms=10.0,
                window_ms=20.0, max_freq=10000000, eps=1e-14):
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples,
                                              shape=nshape, strides=nstrides)

    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]

    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2

    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale

    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram

def graph_specgram(name, samples, sampling_rate):
    plt.plot()

    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(samples, Fs=sampling_rate)

    plt.xlabel('Time')
    plt.title(name)
    plt.ylabel('Frequency')
    plt.show()



arr = ["robin", "sparrow", "starling", "chickadee", "dove", "finch", "flicker", "magpie"]

birdsongs = {}

#./dogsounds.wav
for i in arr:
    bird = []
    for j in range(1, 11):

        file_path = "./" + i + "/" + i + "_" + str(j) + ".wav"
        samples, sampling_rate = librosa.load(file_path, sr = None, mono = True,
                                        offset = 0.0, duration = None)
        graph_specgram(i + " " + str(j), samples, sampling_rate)
    birdsongs[i] = bird

#specto = spectrogram(samples, sampling_rate)

#graph_specgram(samples, sampling_rate)


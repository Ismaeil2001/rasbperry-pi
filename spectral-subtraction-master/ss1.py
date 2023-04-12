import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt

def remove_noise(input_file, noise_file, output_file, noise_output_file):
    # load input file, and stft (Short-time Fourier transform)
    print ('load wav', input_file)
    w, sr = librosa.load(input_file, sr=None, mono=True) # keep native sr (sampling rate) and trans into mono
    s = librosa.stft(w)    # Short-time Fourier transform
    ss = np.abs(s)         # get magnitude
    angle = np.angle(s)    # get phase
    b = np.exp(1.0j * angle) # use this phase information when Inverse Transform

    # load noise file, stft, and get mean
    print ('load wav', noise_file)
    nw, nsr = librosa.load(noise_file, sr=None, mono=True)
    ns = librosa.stft(nw)
    if ns.shape[1] < angle.shape[1]:
        ns = np.pad(ns, ((0, 0), (0, angle.shape[1] - ns.shape[1])), mode='constant')
    nss = np.abs(ns)
    mns = np.mean(nss, axis=1) # get mean

    # subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
    sa = ss - mns.reshape((mns.shape[0],1))  # reshape for broadcast to subtract
    sa0 = sa * b  # apply phase information
    y = librosa.istft(sa0) # back to time domain signal

    # save as a wav file without noise
    scipy.io.wavfile.write(output_file, sr, (y * 32768).astype(np.int16)) # save signed 16-bit WAV format
    print('write wav', output_file)

    # save noise as a wav file
    if ns.shape[1] < angle.shape[1]:
        ns = np.pad(ns, ((0, 0), (0, angle.shape[1] - ns.shape[1])), mode='constant')
    ns = ns * np.exp(1.0j * angle)  # apply phase information
    noise = librosa.istft(ns) # back to time domain signal
    scipy.io.wavfile.write(noise_output_file, sr, (noise * 32768).astype(np.int16)) # save signed 16-bit WAV format
    print('write wav', noise_output_file)

    # plot graphs
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Input, Noise, and Output Signals')

    axs[0].plot(w)
    axs[0].set_title('Input Signal')

    axs[1].plot(nw)
    axs[1].set_title('Noise Signal')

    axs[2].plot(y)
    axs[2].set_title('Output Signal')

    plt.show()

# example usage
input_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/BabyElephantWalk.wav'
noise_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/noise_short.wav'
output_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/BabyElephantWalk_without_noise.wav'
noise_output_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/noise_from_BabyElephantWalk'

remove_noise(input_file, noise_file, output_file, noise_output_file)
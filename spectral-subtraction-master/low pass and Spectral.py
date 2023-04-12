import numpy as np
import scipy
import librosa


def remove_noise(input_file, noise_file, output_file, noise_output_file, cutoff_freq):
    # load input file, and stft (Short-time Fourier transform)
    print('load wav', input_file)
    w, sr = librosa.load(input_file, sr=None, mono=True)  # keep native sr (sampling rate) and trans into mono
    s = librosa.stft(w)  # Short-time Fourier transform
    ss = np.abs(s)  # get magnitude
    angle = np.angle(s)  # get phase
    b = np.exp(1.0j * angle)  # use this phase information when Inverse Transform

    # load noise file, stft, and get mean
    print('load wav', noise_file)
    nw, nsr = librosa.load(noise_file, sr=None, mono=True)
    ns = librosa.stft(nw)
    if ns.shape[1] < angle.shape[1]:
        ns = np.pad(ns, ((0, 0), (0, angle.shape[1] - ns.shape[1])), mode='constant')
    nss = np.abs(ns)
    mns = np.mean(nss, axis=1)  # get mean

    # subtract noise spectral mean from input spectral, and ist ft (Inverse Short-Time Fourier Transform)
    sa = ss - mns.reshape((mns.shape[0], 1))  # reshape for broadcast to subtract
    sa0 = sa * b  # apply phase information

    # apply low pass filter
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
    sa0 = scipy.signal.lfilter(b, a, sa0)

    y = librosa.istft(sa0)  # back to time domain signal

    # save as a wav file without noise
    scipy.io.wavfile.write(output_file, sr, (y * 32768).astype(np.int16))  # save signed 16-bit WAV format
    print('write wav', output_file)

    # save noise as a wav file
    if ns.shape[1] < angle.shape[1]:
        ns = np.pad(ns, ((0, 0), (0, angle.shape[1] - ns.shape[1])), mode='constant')
    ns = ns * np.exp(1.0j * angle)  # apply phase information
    noise = librosa.istft(ns)  # back to time domain signal
    scipy.io.wavfile.write(noise_output_file, sr, (noise * 32768).astype(np.int16))  # save signed 16-bit WAV format
    print('write wav', noise_output_file)


# example usage
input_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/BabyElephantWalk.wav'
noise_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/noise_short.wav'
output_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/BabyElephantWalk_without_noise1.wav'
noise_output_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/noise_from_BabyElephantWalk1.wav'
cutoff_freq = 5000

remove_noise(input_file, noise_file, output_file, noise_output_file, cutoff_freq)

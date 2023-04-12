import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt


def butter_lowpass(cutoff, fs, order=5):
    # Calculate the Nyquist frequency
    nyq = 0.5 * fs
    # Calculate the filter coefficients using Butterworth filter design
    sos = butter(order, cutoff / nyq, output='sos', btype='lowpass')
    return sos


def remove_noise(input_file, output_file, method='low_pass', cutoff_freq=3000, noise_file=None):
    y, sr = librosa.load(input_file, sr=44100)
    # load input file, and stft (Short-time Fourier transform)
    print('Loading input file:', input_file)
    w, sr = librosa.load(input_file, sr=None, mono=True)  # keep native sr (sampling rate) and trans into mono
    s = librosa.stft(w)  # Short-time Fourier transform
    ss = np.abs(s)  # get magnitude
    angle = np.angle(s)  # get phase

    # take average magnitude along time axis to get a 1-D array for b
    avg_mag = np.mean(ss, axis=1)
    b = np.exp(1.0j * angle)  # use this phase information when Inverse Transform

    # apply noise reduction method
    if method == 'low_pass':
        # Filter coefficients
        nyquist_freq = 0.5 * sr
        normal_cutoff_freq = cutoff_freq / nyquist_freq
        b, a = butter(5, normal_cutoff_freq, btype='low', analog=False)

        # Apply filter
        y = filtfilt(b, a, y)
    elif method == 'spectral_subtraction':
        if noise_file is None:
            print('Error: Noise file required for spectral subtraction method.')
            return
        print('Using spectral subtraction method with noise file:', noise_file)
        # load noise file, stft, and get mean
        nw, nsr = librosa.load(noise_file, sr=None, mono=True)
        ns = librosa.stft(nw)
        if ns.shape[1] < angle.shape[1]:
            ns = np.pad(ns, ((0, 0), (0, angle.shape[1] - ns.shape[1])), mode='constant')
        nss = np.abs(ns)
        mns = np.mean(nss, axis=1)  # get mean
        # subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
        sa = ss - mns.reshape((mns.shape[0], 1))  # reshape for broadcast to subtract
        sa0 = sa * b  # apply phase information
    else:
        print('Error: Invalid method selected. Valid options: low_pass, spectral_subtraction')
        return

    # back to time domain signal
    y = librosa.istft(sa0)

    # save as a wav file without noise
    print('Writing output file:', output_file)
    librosa.output.write(output_file, y, sr)


# example usage
input_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/BabyElephantWalk.wav'
output_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/noise_from_BabyElephantWalk4.wav'
noise_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/noise_short.wav'

# prompt user to choose between low-pass and spectral subtraction filter
method = input('Choose noise reduction method: (1) low-pass filter or (2) spectral subtraction filter? Enter 1 or 2: ')
if method == '1':
    cutoff_freq = int(input('Enter cutoff frequency (in Hz): '))
    remove_noise(input_file, output_file, method='low_pass', cutoff_freq=cutoff_freq)
elif method == '2':
    remove_noise(input_file, output_file, noise_file=noise_file, method='spectral_subtraction')
else:
    print('Error: Invalid input. Please enter 1 or 2.')

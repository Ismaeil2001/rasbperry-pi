import numpy as np
import scipy
import librosa
import sounddevice as sd
def remove_noise_from_microphone(duration, output_file, noise_output_file, sr=16000, block_length=1024, hop_length=512):
    # Open the microphone stream
    stream = sd.InputStream(channels=1, samplerate=sr, blocksize=block_length)

    # Load the noise file and compute its mean
    print("Loading noise file...")
    noise_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/noise_short.wav'
    noise, _ = librosa.load(noise_file, sr=sr, mono=True)
    noise_stft = librosa.stft(noise, n_fft=block_length, hop_length=hop_length)
    noise_mean = np.mean(np.abs(noise_stft), axis=1)

    # Initialize the output stream
    print("Recording...")
    frames = []
    with stream:
        for i in range(int(sr * duration / block_length)):
            block = stream.read(block_length)[0]
            frames.append(block)

    # Combine all recorded blocks
    input_signal = np.concatenate(frames)

    # Compute the STFT of the input signal and subtract the noise mean
    input_stft = librosa.stft(input_signal, n_fft=block_length, hop_length=hop_length)
    input_spec = np.abs(input_stft)
    input_phase = np.angle(input_stft)
    noise_mean = noise_mean.reshape((-1, 1))
    cleaned_spec = input_spec - noise_mean
    cleaned_spec[cleaned_spec < 0] = 0

    # Reconstruct the cleaned signal
    cleaned_stft = cleaned_spec * np.exp(1j * input_phase)
    cleaned_signal = librosa.istft(cleaned_stft, hop_length=hop_length)

    # Write the cleaned signal and the noise to disk
    scipy.io.wavfile.write(output_file, sr, (cleaned_signal * 32768).astype(np.int16))
    print('write wav', output_file)
    noise_stft = noise_stft[:, :cleaned_stft.shape[1]]
    noise_signal = librosa.istft(noise_stft, hop_length=hop_length)
    scipy.io.wavfile.write(noise_output_file, sr, (noise_signal * 32768).astype(np.int16))
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
duration = 5
output_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/output_file.wav'
noise_output_file = '/Users/Mubby/Downloads/spectral-subtraction-master/samples/noise_output_file.wav'
remove_noise_from_microphone(duration, output_file, noise_output_file)

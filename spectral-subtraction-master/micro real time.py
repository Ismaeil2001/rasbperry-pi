import numpy as np
import sounddevice as sd
import librosa
import warnings

# Set the block length (in number of samples)
block_length = 1024

# Set the threshold for noise reduction
threshold = 5

# Load the noise sample
noise_file = 'noise_short.wav'
try:
    noise, sr = librosa.load(noise_file, sr=None, mono=True)
except:
    warnings.warn('PySoundFile failed. Trying audioread instead.')
    noise, sr = librosa.load(noise_file, sr=None, mono=True, method='audioread')

# Define the noise reduction function
def reduce_noise(block, noise, threshold):
    # Compute the noise spectrum
    noise_spectrum = np.abs(np.fft.fft(noise))
    # Compute the input block spectrum
    block_spectrum = np.abs(np.fft.fft(block))
    # Compute the spectral subtraction
    subtracted_spectrum = block_spectrum - threshold * noise_spectrum
    # Set negative values to 0
    subtracted_spectrum[subtracted_spectrum < 0] = 0
    # Compute the cleaned block spectrum
    cleaned_spectrum = subtracted_spectrum * np.exp(1j * np.angle(np.fft.fft(block)))
    # Compute the cleaned block
    cleaned_block = np.fft.ifft(cleaned_spectrum).real
    return cleaned_block

# Define the noise removal function for microphone input
def remove_noise_from_microphone(duration):
    # Start the input stream
    input_stream = sd.InputStream(channels=1, blocksize=block_length, samplerate=sr)
    input_stream.start()
    # Start the output stream
    output_stream = sd.OutputStream(channels=1, blocksize=block_length, samplerate=sr)
    output_stream.start()
    # Initialize the buffer
    buffer = np.zeros((block_length,), dtype=np.float32)
    # Initialize the number of blocks read
    num_blocks = 0
    # Loop over the blocks
    while num_blocks < int(sr * duration / block_length):
        # Read the block from the input stream
        input_block, overflowed = input_stream.read(block_length)
        # If there was an overflow, print a warning message
        if overflowed:
            warnings.warn('Input overflowed')
        # Compute the cleaned block
        cleaned_block = reduce_noise(input_block, noise, threshold)
        # Write the cleaned block to the output stream
        output_stream.write((cleaned_block * 32768).astype(np.int16))
        # Increment the number of blocks read
        num_blocks += 1
    # Stop the streams
    input_stream.stop()
    output_stream.stop()
    # Close the streams
    input_stream.close()
    output_stream.close()

# Call the noise removal function with a duration of 10 seconds
remove_noise_from_microphone(duration=10)

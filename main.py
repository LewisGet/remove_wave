import numpy as np
import librosa
from scipy.fftpack import fft
from pydub import AudioSegment
import os
import glob

sampling_rate = 48000
output_dir = "split_audio"
noise_folder = "sample"
only_export_full_version = True

def calculate_similarity(wave1, wave2):
    wave1 = np.array(wave1)
    wave2 = np.array(wave2)

    len1 = len(wave1)
    len2 = len(wave2)

    length = min(len1, len2)

    try:
        fft1 = fft(wave1[:length])
        fft2 = fft(wave2[:length])
    except Exception as e:
        print(e)
        return 0

    print("fft working")

    correlation = np.corrcoef(fft1, fft2)[0][1]

    return correlation

def audio_format(audio):
    audio = audio.set_frame_rate(sampling_rate)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)

    return audio

def audio_format_array(audio):
    return audio_format(audio).get_array_of_samples()

def array_to_audio_segment(array, sample_rate=44100, sample_width=2):
    # Convert the array to a NumPy array if necessary
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    # Add a channel dimension if the array is one-dimensional
    if array.ndim == 1:
        array = array[:, np.newaxis]

    # Normalize the array values to the range [-1, 1]
    array = array / np.max(np.abs(array))

    # Convert the array to a bytes object
    array = (array * 2**15).astype(np.int16)
    audio_bytes = array.tobytes()

    # Create the AudioSegment object
    audio_segment = AudioSegment(
        audio_bytes, 
        frame_rate=sample_rate, 
        sample_width=sample_width,
        channels=array.shape[1]
    )

    return audio_segment

def add_silent(audio):
    # 0.1 seconds in milliseconds
    pause_duration = 0.3 * 1000
    pause = AudioSegment.silent(duration=pause_duration)

    return audio + pause

def split_audio_by_noise(audio, output_dir):
    # Create output directories if they don't already exist
    if not os.path.exists(output_dir + "/noise"):
        os.makedirs(output_dir + "/noise")
    if not os.path.exists(output_dir + "/clean"):
        os.makedirs(output_dir + "/clean")

    new_audio = AudioSegment.silent(duration=1)

    segment_length = None
    noise_samples = []
    audio_data = audio_format_array(audio)

    for i in glob.glob(os.path.join(".", noise_folder, "*.wav")):
        noise_data = audio_format(AudioSegment.from_file(i))
        noise_data = add_silent(noise_data)
        noise_data = audio_format_array(noise_data)
        noise_samples.append(noise_data)

        size = len(noise_data)

    if segment_length is None or size < segment_length:
        segment_length = size

    for i in range(0, len(audio_data), segment_length):
        start_time = i * segment_length
        end_time = start_time + segment_length
        segment = audio[start_time:end_time]

        noise_present = False
        for noise in noise_samples:
            if calculate_similarity(segment, noise) > 0.8:
                noise_present = True
                break

        segment = array_to_audio_segment(segment, audio.frame_rate, audio.sample_width)

        if not only_export_full_version:
            if noise_present:
                segment.export(os.path.join(output_dir, "noise", "segment_{}.wav".format(i)), format="wav")
            else:
                segment.export(os.path.join(output_dir, "clean", "segment_{}.wav".format(i)), format="wav")

        if not noise_present:
            new_audio = new_audio + audio[start_time:start_time + segment_length]

    new_audio.export(os.path.join(output_dir, "clean", "clean_version.wav".format(i)), format="wav")

audio = AudioSegment.from_file("test.wav")
split_audio_by_noise(audio, output_dir)

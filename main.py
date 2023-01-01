import numpy as np
import librosa
from scipy.fftpack import fft
from pydub import AudioSegment
import os
import glob

sampling_rate = 48000
output_dir = "split_audio"
noise_folder = "sample"

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

def split_audio_by_noise(audio, noise_samples, segment_length, output_dir):
    # Create output directories if they don't already exist
    if not os.path.exists(output_dir + "/noise"):
        os.makedirs(output_dir + "/noise")
    if not os.path.exists(output_dir + "/clean"):
        os.makedirs(output_dir + "/clean")
    match_noise_samples = []

    for noise in noise_samples:
        match_noise_samples.append(audio_format_array(noise))
    
    for i in range(0, len(audio), segment_length):
        start_time = i * segment_length
        end_time = start_time + (segment_length * 2)
        segment = audio[start_time:end_time]
        match_format = audio_format_array(segment)

        noise_present = False
        for noise in match_noise_samples:
            if calculate_similarity(match_format, noise) > 0.8:
                noise_present = True
                break
        
        if noise_present:
            segment.export(os.path.join(output_dir, "noise", "segment_{}.wav".format(i)), format="wav")
        else:
            segment.export(os.path.join(output_dir, "clean", "segment_{}.wav".format(i)), format="wav")


def add_silent(audio):
    # 0.1 seconds in milliseconds
    pause_duration = 0.1 * 1000
    pause = AudioSegment.silent(duration=pause_duration)

    return audio + pause

audio = AudioSegment.from_file("test.wav")

segment_length = None
noise_samples = []

for i in glob.glob(os.path.join(".", noise_folder, "*.wav")):
    noise_data = audio_format(AudioSegment.from_file(i))
    noise_samples.append(add_silent(noise_data))

    size = len(noise_data)

    if segment_length is None or size < segment_length:
        segment_length = size

segment_length = int(segment_length // 2)
split_audio_by_noise(audio, noise_samples, segment_length, output_dir)

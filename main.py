import numpy as np
import librosa
from scipy.fftpack import fft
from pydub import AudioSegment
import os
import glob

sampling_rate = 48000
saving_datatype = "float32"
output_dir = "split_audio"
noise_folder = "sample"
cache_folder = "cache"
rebuild_cache = False
only_export_full_version = False

def calculate_similarity(wave1, wave2):
    wave1 = np.array(wave1)
    wave2 = np.array(wave2)

    len1 = len(wave1)
    len2 = len(wave2)

    length = min(len1, len2)

    if length is 0:
        return 0

    try:
        fft1 = fft(wave1[:length])
        fft2 = fft(wave2[:length])
    except Exception as e:
        print(e)
        return 0

    correlation = np.corrcoef(fft1, fft2)[0][1]

    return correlation

def audio_format(audio):
    audio = audio.set_frame_rate(sampling_rate)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)

    return audio

def audio_format_array(audio):
    return audio_format(audio).get_array_of_samples()

def split_audio_by_noise(target_audio_path, output_dir):
    # Create output directories if they don't already exist
    if not os.path.exists(output_dir + "/noise"):
        os.makedirs(output_dir + "/noise")
    if not os.path.exists(output_dir + "/clean"):
        os.makedirs(output_dir + "/clean")

    new_audio = []
    noise_samples = []
    segment_length = None
    audio_data, _ = load_wav(os.path.join(cache_folder, target_audio_path))

    for i in glob.glob(os.path.join(".", cache_folder, noise_folder, "*.wav")):
        noise_data, _ = load_wav(i)
        noise_samples.append(noise_data)

        size = len(noise_data)

        offset_noise_shape = list(noise_data.shape)
        offset_noise_shape[-1] = int(offset_noise_shape[-1] // 2)
        offset_noise_data = np.array(np.zeros(offset_noise_shape)).tolist()
        offset_noise_data = list(offset_noise_data + noise_data.tolist())
        
        noise_samples.append(offset_noise_data)

        if segment_length is None or size < segment_length:
            segment_length = size

    for i in range(0, len(audio_data), segment_length):
        start_time = i
        end_time = start_time + segment_length
        segment = audio_data[start_time:end_time]

        noise_present = False
        for noise in noise_samples:
            if calculate_similarity(segment, noise) > 0.8:
                noise_present = True
                break

        if not only_export_full_version:
            if noise_present:
                write_wav(os.path.join(output_dir, "noise", "segment_{}.wav".format(i)), segment)
            else:
                write_wav(os.path.join(output_dir, "clean", "segment_{}.wav".format(i)), segment)

        if not noise_present:
            if isinstance(segment, np.ndarray):
                segment = segment.tolist()

            if isinstance(new_audio, np.ndarray):
                new_audio = new_audio.tolist()

            new_audio = new_audio + segment
    
    write_wav(os.path.join(output_dir, "clean", "clean_version.wav"), new_audio)

def load_wav(path):
    return librosa.load(path, sr=None)

def write_wav(path, data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    librosa.output.write_wav(path, data.astype(saving_datatype), sampling_rate)

def build_cache(target_path):
    audio = AudioSegment.from_file(target_path)
    name = os.path.basename(target_path)
    folders = [cache_folder] + os.path.dirname(target_path).split(os.sep)

    for level, dir_name in enumerate(folders):
        if dir_name == ".":
            continue

        path = os.path.join(*folders[0:level + 1])

        if not os.path.exists(path):
            os.makedirs(path)

    new_path = os.path.join(*folders, name)

    audio_format(audio).export(new_path, format="wav")

target_audio_path = "test.wav"

if rebuild_cache:
    for i in glob.glob(os.path.join(".", noise_folder, "*.wav")) + [target_audio_path]:
        build_cache(i)

split_audio_by_noise(target_audio_path, output_dir)

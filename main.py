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

def calculate_similarity(wave1, fft2):
    wave1 = np.array(wave1)

    if len(wave1) is 0:
        return 0

    try:
        fft1 = fft(wave1)
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

def split_audio_by_noise(target_audio_path, output_dir, output_name="clean_version.wav"):
    # Create output directories if they don't already exist
    if not os.path.exists(output_dir + "/noise"):
        os.makedirs(output_dir + "/noise")
    if not os.path.exists(output_dir + "/clean"):
        os.makedirs(output_dir + "/clean")

    new_audio = []
    noise_samples = []
    noise_len = []
    audio_data, _ = load_wav(os.path.join(cache_folder, target_audio_path))

    for i in glob.glob(os.path.join(".", cache_folder, noise_folder, "*.wav")):
        noise_data, _ = load_wav(i)

        if len(noise_data) == 0:
            continue

        noise_fft = fft(np.array(noise_data))
        noise_len.append(len(noise_data))
        noise_samples.append(noise_fft)

    total_execute_times = len(audio_data)

    i = 0
    while i < total_execute_times:
        print("update:", i, "/", total_execute_times)
        print("process %d" % int(i//(total_execute_times/100) + 1))

        match_length = 0
        noise_present = False
        for noise, noise_length in zip(noise_samples, noise_len):
            start_time, end_time = i, i + noise_length
            audio_sample = audio_data[start_time:end_time]

            if calculate_similarity(audio_sample, noise) > 0.8:
                match_length = noise_length
                noise_present = True
                break

        if noise_present:
            i += match_length
        else:
            clean_segment = audio_data[i:i+1]

            if isinstance(clean_segment, np.ndarray):
                clean_segment = clean_segment.tolist()

            if isinstance(new_audio, np.ndarray):
                new_audio = new_audio.tolist()

            new_audio = new_audio + clean_segment

            i += 1
    
    write_wav(os.path.join(output_dir, "clean", output_name), new_audio)

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

if __name__ == "__main__":
    target_audio_path = "unpreprocess.wav"

    if rebuild_cache:
        for i in glob.glob(os.path.join(".", noise_folder, "*.wav")) + [target_audio_path]:
            build_cache(i)

    split_audio_by_noise(target_audio_path, output_dir)

from main import *
import os
import glob
import sys

if __name__ == "__main__":
    rebuild_cache = True
    only_export_full_version = True

    target_audio_path = sys.argv[1]

    if rebuild_cache:
        for i in [target_audio_path]:
            build_cache(i)

    split_audio_by_noise(target_audio_path, output_dir, sys.argv[1])

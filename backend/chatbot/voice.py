import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

# Allowlist the required classes for safe unpickling
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig])

# Initialize the TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Path to the specific .wav file
wav_file = "data/tts/Michael_2.wav"

if not os.path.isfile(wav_file):
    print(f"No such file: {wav_file}")
else:
    print(f"Processing: {wav_file}")
    tts.tts_to_file(
        text="With this we're testing if the GPU processor works for creating the audio.",
        file_path=os.path.join("data", "tts", f"output_{os.path.basename(wav_file)}"),
        speaker_wav=wav_file,
        language="en"
    )
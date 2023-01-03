import os
import librosa
import librosa.display
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from sparse_image_warp import sparse_image_warp
matplotlib.use('TkAgg')

import requests 

import torch
import torch.nn as nn

"""
https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
https://github.com/tuanio/conformer-rnnt/blob/main/model/spec_augment.py
https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/spectr_augment.py
https://github.com/DemisEom/SpecAugment/blob/7f1435963b37ac8f9e4de9e44d754ecc41eaba85/SpecAugment/spec_augment_pytorch.py
https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
https://github.com/pyyush/SpecAugment/blob/master/augment.py
https://pytorch.org/audio/master/tutorials/audio_feature_augmentation_tutorial.html
https://github.com/zcaceres/spec_augment

"""

__all__ = ["SpecAugment"]

_SAMPLE_DIR = "_assets"

SAMPLE_WAV_SPEECH_URL = "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav"  # noqa: E501
SAMPLE_WAV_SPEECH_PATH = os.path.join(_SAMPLE_DIR, "sound.wav")

os.makedirs(_SAMPLE_DIR, exist_ok=True)

def _fetch_data():
    uri = [
        (SAMPLE_WAV_SPEECH_URL, SAMPLE_WAV_SPEECH_PATH),
    ]
    print('Downloading wav files...')
    for url, path in uri:
        with open(path, "wb") as file_:
            file_.write(requests.get(url).content)

_fetch_data()

def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    # assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]
    # assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


class SpecAugment(nn.Module):
    """Spec augmentation Calculation Function.
        'SpecAugment' have 3 steps for audio data augmentation.
        first step is time warping using Tensorflow's image_sparse_warp function.
        Second step is frequency masking, last step is time masking.
        # Arguments:
        mel_spectrogram(numpy array): audio file path of you want to warping and masking.
        time_warping_para(float): Augmentation parameter, "time warp parameter W".
            If none, default = 80 for LibriSpeech.
        frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
            If none, default = 100 for LibriSpeech.
        time_masking_para(float): Augmentation parameter, "time mask parameter T"
            If none, default = 27 for LibriSpeech.
        frequency_mask_num(float): number of frequency masking lines, "m_F".
            If none, default = 1 for LibriSpeech.
        time_mask_num(float): number of time masking lines, "m_T".
            If none, default = 1 for LibriSpeech.
        # Returns
        mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    def __init__(self, time_warping_para=40, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=1, time_mask_num=1):

        self.time_warping_para = time_warping_para
        self.frequency_masking_para = frequency_masking_para
        self.time_masking_para = time_masking_para
        self.frequency_mask_num = frequency_mask_num
        self.time_mask_num = time_mask_num
        
    def __call__(self, mel_spectrogram):
        v = mel_spectrogram.shape[1]
        tau = mel_spectrogram.shape[2]

        # Step 1 : Time warping
        warped_mel_spectrogram = time_warp(mel_spectrogram, W=self.time_warping_para)

        # Step 2 : Frequency masking
        for i in range(self.frequency_mask_num):
            f = np.random.uniform(low=0.0, high=self.frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v-f)
            warped_mel_spectrogram[:, f0:f0+f, :] = 0

        # Step 3 : Time masking
        for i in range(self.time_mask_num):
            t = np.random.uniform(low=0.0, high=self.time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau-t)
            warped_mel_spectrogram[:, :, t0:t0+t] = 0

        return warped_mel_spectrogram

    def __repr__(self):
        return f"{self.__class__.__name__}()"

def visualization_spectrogram(mel_spectrogram, title, block=True):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
    
if __name__ == '__main__':
    
    torch.random.manual_seed(4)
    
    audio_path = os.path.join(_SAMPLE_DIR, 'sound.wav')
    audio, sampling_rate = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)

    shape = mel_spectrogram.shape
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
    mel_spectrogram = torch.from_numpy(mel_spectrogram)
    
    # Show Raw mel-spectrogram
    visualization_spectrogram(mel_spectrogram=mel_spectrogram, title='Raw mel spectrogram', block=False)
    
    augmentor = SpecAugment()
    warped_masked_spectrogram = augmentor(mel_spectrogram)
    
    # Show Time Warped & Masked mel-spectrogram
    visualization_spectrogram(mel_spectrogram=warped_masked_spectrogram, title="pytorch Warped & Masked Spectrogram")

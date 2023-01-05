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

__all__ = ["SpecAugment", "MixUp", "CutMix"]

_SAMPLE_DIR = "_assets"

SAMPLE_WAV_SPEECH_URLS = ["https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav", 
                          "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars3.wav"]
SAMPLE_WAV_SPEECH_PATHS = [os.path.join(_SAMPLE_DIR, "sample1.wav"), os.path.join(_SAMPLE_DIR, "sample2.wav")]

os.makedirs(_SAMPLE_DIR, exist_ok=True)

def _fetch_data():
    uris = [
        (SAMPLE_WAV_SPEECH_URLS[0], SAMPLE_WAV_SPEECH_PATHS[0]), 
        (SAMPLE_WAV_SPEECH_URLS[1], SAMPLE_WAV_SPEECH_PATHS[1])
    ]
    if all([os.path.exists(path) for _, path in uris]):
        return
    
    print('Downloading wav files...')
    for url, path in uris:
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

class MixUp(nn.Module):
    def __init__(self, batch_size):
        self.mixup_lambda = torch.from_numpy(self.get_mix_lambda(0.5, batch_size))
        
    def __call__(self, x):
        """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes (1, 3, 5, ...).
        Args:
            x: (batch_size * 2, ...)
            mixup_lambda: (batch_size * 2,)
            Returns:
            out: (batch_size, ...)
        """
        out = (x[0 :: 2].transpose(0, -1) * self.mixup_lambda[0 :: 2] + \
            x[1 :: 2].transpose(0, -1) * self.mixup_lambda[1 :: 2]).transpose(0, -1)
        return out
    
    def get_mix_lambda(self, mixup_alpha, batch_size):
        # Draw sample from a Beta distribution
        mixup_lambdas = [np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in range(batch_size)]
        return np.array(mixup_lambdas).astype(np.float32)

def visualization_spectrogram(mel_spectrogram, index, title="", block=True):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[index, :, :], ref=np.max), x_axis='time', y_axis='mel', fmax=8000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
    
if __name__ == '__main__':
    
    torch.random.manual_seed(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    display = False
    
    augmentation_to_apply = ['mixup']
    
    specs = []
    lengths = []
    sr = []
    for audio_path in SAMPLE_WAV_SPEECH_PATHS:
        audio, sampling_rate = librosa.load(audio_path)
        lengths.append(len(audio))
        sr.append(sampling_rate)
        print(f'audio: {audio_path} / shape: {audio.shape} / sample rate: {sampling_rate}')
        mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                        sr=sampling_rate,
                                                        n_mels=256,
                                                        hop_length=128,
                                                        fmax=8000)
        specs.append(np.expand_dims(mel_spectrogram, axis=0))
    mel_spectrogram = np.vstack(specs)

    shape = mel_spectrogram.shape
    mel_spectrogram = torch.from_numpy(mel_spectrogram)
    
    if display:
        # Show Raw mel-spectrogram
        visualization_spectrogram(mel_spectrogram, 0, title='Raw mel spectrogram-1', block=False)
        visualization_spectrogram(mel_spectrogram, 1, title='Raw mel spectrogram-2', block=False)
    
    batch_size = 1
    batch_size = 2 * batch_size if 'mixup' in augmentation_to_apply else batch_size
    augmentor = torch.nn.Sequential(MixUp(batch_size))
    warped_masked_spectrogram = augmentor(mel_spectrogram)
    
    if display:
        # Show Time Warped & Masked mel-spectrogram
        visualization_spectrogram(warped_masked_spectrogram, index = 0, title="pytorch Warped & Masked Spectrogram")
    
    y_recov = librosa.feature.inverse.mel_to_audio(np.array(mel_spectrogram[0]), sr=sr[0])
    y_recov = (y_recov * 32768).astype(np.int16)
    print(y_recov.shape)
    y_augm = librosa.feature.inverse.mel_to_audio(np.array(warped_masked_spectrogram[0]), sr=sr[0])
    y_augm = (y_augm * 32768).astype(np.int16)
    print(y_augm.shape)
    #assert librosa.util.valid_audio(y_augm)
    np.save('raw_recovered', y_recov)
    np.save('raw_augmented', y_augm)

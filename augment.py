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

# SpecAugment pdf: https://arxiv.org/pdf/1904.08779.pdf
# https://github.com/DemisEom/SpecAugment/blob/7f1435963b37ac8f9e4de9e44d754ecc41eaba85/tests/spec_augment_test_pytorch.py#L64
# https://www.researchgate.net/publication/319534366_A_Comparison_on_Audio_Signal_Preprocessing_Methods_for_Deep_Neural_Networks_on_Music_Tagging
# https://colab.research.google.com/drive/1kKm0SVhC4v7SfbBeLfBn9RF-xO_qT8rq#scrollTo=cPouP_kiOixn
# mel-spectrogram: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
# https://judy-son.tistory.com/6
# specaugment: https://github.com/pyyush/SpecAugment
# https://pypi.org/project/spec-augment/
# https://github.com/TeaPoly/SpectrumAugmenter/blob/main/spectrum_augmenter.py
# https://www.kaggle.com/code/yash612/simple-audio-augmentation

"""
    def _get_mel_spectrogram(self, linear_spectra):
        # (time, freq, channels)
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        # (time, channels x freq)
        return mel_feat
"""

_SAMPLE_DIR = "_assets"
SAMPLE_WAV_SPEECH_URLS = ["https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav", 
                          "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars3.wav", 
                          "https://dummy.wav"]
SAMPLE_WAV_SPEECH_PATHS = [os.path.join(_SAMPLE_DIR, "sample1.wav"), os.path.join(_SAMPLE_DIR, "sample2.wav"), os.path.join(_SAMPLE_DIR, "sample3.wav")]
os.makedirs(_SAMPLE_DIR, exist_ok=True)

def _fetch_data():
    uris = [
        (SAMPLE_WAV_SPEECH_URLS[0], SAMPLE_WAV_SPEECH_PATHS[0]), 
        (SAMPLE_WAV_SPEECH_URLS[1], SAMPLE_WAV_SPEECH_PATHS[1]),
        (SAMPLE_WAV_SPEECH_URLS[2], SAMPLE_WAV_SPEECH_PATHS[2]),
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

class Augmentor:
  def __init__(self, list_enabled):
    self.list_enabled = list_enabled
    
  def is_enabled(self, aug_name):
    return aug_name in self.list_enabled
   
  def do_specaugment(self, mel_spectrogram, time_warping_para=40, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=1, time_mask_num=1):

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

    c = mel_spectrogram.shape[0]
    bs = mel_spectrogram.shape[1]
    v = mel_spectrogram.shape[2]
    tau = mel_spectrogram.shape[3]

    # Step 1 : Time warping
    specs = []
    for i in range(c):
        warped_mel_spectrogram = time_warp(mel_spectrogram[i], W=time_warping_para)
        specs.append(warped_mel_spectrogram.unsqueeze(0))
    
    warped_mel_spectrogram = torch.cat(specs, dim=0)   

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v-f)
        warped_mel_spectrogram[:, :, f0:f0+f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau-t)
        warped_mel_spectrogram[:, :, :, t0:t0+t] = 0

    return warped_mel_spectrogram

  def do_mixup(self, x, x1, y=None, y1=None, a=0.5, b=0.5, test=False):

      """Mixup (x, y) with (x1, y1).
      Args:
          x: (seq_len, nb_ch, nb_mel_bins)
          x1: (seq_len, nb_ch, nb_mel_bins)
          y: (seq_len, nb_track_dummy, nb_axis, nb_class)
          y1: (seq_len, nb_track_dummy, nb_axis, nb_class)
      Returns:
          out: (batch_size,)
      """

      a1 = np.random.beta(a, b)
      if test or (np.random.rand() < 0.8 and np.abs(a1 - 0.5) > 0.2):
        x = a1 * x + (1 - a1) * x1
        if y is not None and y1 is not None:
          y = a1 * y + (1 - a1) * y1

      return (x, y) if y is not None and y1 is not None else x

def visualization_spectrogram(mel_spectrogram, title="", block=True):

    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=8000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)

if __name__ == '__main__':

    torch.random.manual_seed(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    display = True

    mono, sampling_rate = librosa.load(SAMPLE_WAV_SPEECH_PATHS[0])
    stereo_audio, sampling_rate = librosa.load(SAMPLE_WAV_SPEECH_PATHS[2])
    mono_flip = np.flip(mono)
    print(f'audio: {SAMPLE_WAV_SPEECH_PATHS[0]} / shape: {mono.shape} / sample rate: {sampling_rate}')

    mel1 = librosa.feature.melspectrogram(y=mono,
                                          sr=sampling_rate,
                                          n_mels=256,
                                          hop_length=128,
                                          fmax=8000)

    mel2 = librosa.feature.melspectrogram(y=mono_flip,
                                          sr=sampling_rate,
                                          n_mels=256,
                                          hop_length=128,
                                          fmax=8000)

    C = 2
    B = 1
    mel_stereo = []
    for ch_cnt in range(C):
        tmp_mel = librosa.feature.melspectrogram(y=stereo_audio[ch_cnt::C],
                                            sr=sampling_rate,
                                            n_mels=256,
                                            hop_length=128,
                                            fmax=8000)
        tmp_mel = torch.tensor(tmp_mel).to(device).view(1, B, tmp_mel.shape[0], tmp_mel.shape[1])
        mel_stereo.append(tmp_mel)
    mel_stereo = torch.cat(mel_stereo, dim=0)

    mel1 = torch.tensor(mel1).to(device).view(1, B, mel1.shape[0], mel1.shape[1])
    mel2 = torch.tensor(mel2).to(device).view(1, B, mel2.shape[0], mel2.shape[1])
    # [C, B, F, T] = [n_channels, batch_size, n_mel_bins, n_timesteps]
    print(mel1.shape)
    print(mel2.shape)
    print(mel_stereo.shape)

    if display:
        # Show Raw mel-spectrogram
        visualization_spectrogram(mel1[0][0], title='Raw mel spectrogram-1', block=False)
        visualization_spectrogram(mel2[0][0], title='Raw mel spectrogram-2', block=False)

    augmentor = Augmentor(['specaugment', 'mixup'])
    if augmentor.is_enabled('mixup'):
      warped_mel = augmentor.do_mixup(mel1, mel2, test=True)

    if augmentor.is_enabled('specaugment'):
      warped_mel = augmentor.do_specaugment(warped_mel)

    if display:
        # Show Time Warped & Masked mel-spectrogram
        visualization_spectrogram(warped_mel[0][0], title="pytorch Warped & Masked Spectrogram")

    """
    y_recov = librosa.feature.inverse.mel_to_audio(np.array(mel_spectrogram[0]), sr=sr[0])
    y_recov = (y_recov * 32768).astype(np.int16)
    print(y_recov.shape)
    y_augm = librosa.feature.inverse.mel_to_audio(np.array(warped_masked_spectrogram[0]), sr=sr[0])
    y_augm = (y_augm * 32768).astype(np.int16)
    print(y_augm.shape)
    #assert librosa.util.valid_audio(y_augm)
    np.save('raw_recovered', y_recov)
    np.save('raw_augmented', y_augm)
    """

# coding: utf-8

# Harmonic Pitch Class Profile extraction

import time
import numpy as np

# import numpy as np
from scipy.io import wavfile
from scipy.sparse import coo_matrix
from scipy.signal import spectrogram, convolve2d
import json
import sys
import librosa
from math import log2
from scipy.io import wavfile
from scipy.fft import fft


def read_audio(file_name):
    try:
        sr, y = wavfile.read(file_name)
    except IOError:
        print(
            "File not found or inappropriate format. \n"
            "Audio file should be in WAV format.\n"
        )
        raise

    # if stereo, average channels
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # normalize
    y = y / np.max(y)
    return y, sr


def calculate_hpcp(audio_path, hop_length=512):
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Extract harmonic pitch classes using librosa's hpcp function
    hpcp = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    hpcp_sum = np.sum(hpcp, axis=1)

    return hpcp_sum


def myhpcp(audio_path, fref):
    """

    recording dataset

    1. normal
    2. fast
    3. reverse normal
    4. reverse fast
    5. each string

    """
    y, sr = librosa.load(audio_path)
    fft_val = np.fft.fft(y)

    fref = [
        16.35,
        17.32,
        18.35,
        19.45,
        20.60,
        21.83,
        23.12,
        24.50,
        25.96,
        27.50,
        29.14,
        30.87,
    ]

    N = len(fft_val)

    # print(N)
    def M(l, fs, fref):
        if l == 0:
            return -1
        return round(12 * log2((fs * l) / (N * fref))) % 12

    pcp = [0 for p in range(12)]  # [0,0,0,0,0,0,0,0,0,0,0,0]
    for p in range(12):
        for l in range((N // 2) - 1):
            temp = M(l, fs=sr, fref=fref)
            # print(f"(p,spectrum)({p},{temp})")
            # time.sleep(0.1)
            if p == temp:  # p = 0...11
                # print(f"(fft)({fft_val[l]})")
                h = abs(fft_val[l]) ** 2  #
                # print(h)
                pcp[p] += h

    """
    EXAMPLE OUTPUT

    this indicate the value of intensity in Pitch Class Profile, starting from C - B. in total of 12 notes

    the sample that send to the algorithm to process is C chord and C chord is consist of 3 notes (triad)
    (root, major, perfect) : (C, E, G)
    so in the output, the highest number of intensity will be C, E, G as labeled below

    [
            0.4563520542813354, -> C        *
            0.002387393810446727, -> C#
            0.001242964139739269, -> D
            0.0004638740418360587, -> D#
            0.08568564686599614, -> E       *
            0.0006450661047309768, -> F
            0.0015144776862368974, -> F#
            0.41702694670510126, -> G       *
            0.003887232844608621, -> G#
            0.013018957229843356, -> A
            0.00878288777947088, -> A#
            0.008992498510654605 -> B
        ]
    
    """

    # Normalize pcp
    pcp_norm = [0 for p in range(12)]
    for p in range(12):
        pcp_norm[p] = pcp[p] / sum(pcp)
    # print("finished pcp")
    # pcp_norm.append(0)
    # print(type(pcp_norm))
    return list(pcp_norm)


def my_enhanced_hpcp(audio_path, fref, pcp_num: int):

    y, sr = librosa.load(audio_path)
    fft_val = np.fft.fft(y)

    N = len(fft_val)

    # print(N)
    def M(l, fs, fref):
        if l == 0:
            return -1
        return round(pcp_num * log2((fs * l) / (N * fref))) % pcp_num

    pcp = [0 for p in range(pcp_num)]  # [0,0,0,0,0,0,0,0,0,0,0,0]
    for p in range(pcp_num):
        for l in range((N // 2) - 1):
            temp = M(l, fs=sr, fref=fref)
            # print(f"(p,spectrum)({p},{temp})")
            # time.sleep(0.1)
            if p == temp:  # p = 0...11
                # print(f"(fft)({fft_val[l]})")
                h = abs(fft_val[l]) ** 2  #
                # print(h)
                pcp[p] += h

    # Normalize pcp
    pcp_norm = [0 for p in range(pcp_num)]
    for p in range(pcp_num):
        pcp_norm[p] = pcp[p] / sum(pcp)
    # print("finished pcp")
    # pcp_norm.append(0)
    # print(type(pcp_norm))
    return list(pcp_norm)

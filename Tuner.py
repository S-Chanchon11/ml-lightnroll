from math import log2
# import librosa
import numpy as np


# def tuner(file):

#     # Load the audio file
#     y, sr = librosa.load(file)

#     # Calculate the pitch
#     pitch, magnitudes = librosa.piptrack(y=y, sr=sr)

#     # Get the index of maximum value in pitch array
#     pitch_index = np.argmax(pitch, axis=0)

#     # Convert the index to pitch values
#     frequencies = librosa.fft_frequencies(sr=sr)
#     pitch_values = frequencies[pitch_index]

#     print(pitch_values)

    # ALTERNATIVE CHOICE

    # Detection = aubio.pitch("yin", 2048, 2048//2, sr)

    # # Empty list to store pitch values
    # pitch_values = []

    # # Iterate through audio frames and compute pitch
    # for frame in range(0, len(y), 2048):
    #     pitch = pDetection(y[frame:frame+2048])[0]
    #     pitch_values.append(pitch)

def pitch_class_profile(audio_path):
    y=1
    sr=2
    # y, sr = librosa.load(audio_path)
    fft_val = np.fft.fft(y)

    N = len(fft_val)

    def M(l, fs, fref):
        if l == 0:
            return -1
        return round(12 * log2((fs * l) / (N * fref))) % 12

    pcp = [0 for p in range(12)] 
    for p in range(12):
        for l in range((N // 2) - 1):

            temp = M(l, fs=sr, fref=261.63)            
            
            if p == temp:  # p = 0...11
                
                h = abs(fft_val[l]) ** 2
                
                pcp[p] += h

    pcp_norm = [0 for p in range(12)]
    for p in range(12):
        pcp_norm[p] = pcp[p] / sum(pcp)

    return list(pcp_norm)


test = 11
file = ''
tuner(file=file)
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from math import log2


class PitchClassProfiler:
    def __init__(self, file_name):
        self.file_name = file_name
        self.read = False

    def _read_file(self):
        self._frequency, self._samples = wavfile.read(self.file_name)
        self.read = True

    def frequency(self):
        if not self.read:
            self._read_file()
        return self._frequency

    def samples(self):
        if not self.read:
            self._read_file()
        return self._samples

    def fourier(self):
        return fft(self.samples())  # Discrete Fourier Transform

    def pcp(self, X):

        fs = self.frequency()
        # print(fs)  # sample rate e.g. 48kHz or 48,000
        # print(X)   # audio time series in ndarray
        # fref = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87]
        # fref = [130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94]
        fref = 130.81

        N = len(X)
        # print(N)
        # assert(N % 2 == 0)

        def M(l, p, fref):
            if l == 0:
                return -1
            return round(12 * log2((fs * l) / (N * fref))) % 12

        pcp = [0 for p in range(12)]

        # print("Computing pcp...")
        for p in range(12):
            for l in range(N // 2):
                # for j in range(len(fref)):
                if p == M(l, p, fref=fref):
                    pcp[p] += abs(X[l]) ** 2
                    # print(f"===={pcp[p]}")

        # Normalize pcp
        pcp_norm = [0 for p in range(12)]
        for p in range(12):
            pcp_norm[p] = pcp[p] / sum(pcp)
        # print("finished pcp")
        # pcp_norm.append(0)
        # print(type(pcp_norm))
        return list(pcp_norm)

    def get_profile(self):
        X = self.fourier()
        return self.pcp(X)



import json
import pickle
import librosa
import numpy as np
from sklearn.model_selection import train_test_split


class Utilities:

    def prepare_data(self,path):
        """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Pitch
        :return y (ndarray): Labels
        :return z (ndarray): Mapping
        """

        with open(path, "r") as fp:
            data = json.load(fp)

        X = np.array(data["pitch"])
        y = np.array(data["labels"])
        z = np.array(data["mapping"])

        return X, y, z
    
    def split_data(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            test_size=0.20, 
            shuffle=True,
            stratify=y
            )
        
        return X_train, X_test, y_train, y_test
    
def load():

    file = '/Users/snow/ml-lightnroll/guitar_chord/All/A/A_DREAM_1.wav'
    y, sr = librosa.load(file)
    np.set_printoptions(threshold=np.inf)
    print(y)
    print(sr)

    pitch, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Get the index of maximum value in pitch array
    pitch_index = np.argmax(pitch, axis=0)

    # Convert the index to pitch values
    frequencies = librosa.fft_frequencies(sr=sr)
    pitch_values = frequencies[pitch_index]

    print(pitch_values)

if __name__ == '__main__':
    load()

    




import json
import pickle
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
            shuffle=True
            )
        
        return X_train, X_test, y_train, y_test


    def save_model(self, model, path):

        with open(path, "wb") as file:
            
            pickle.dump(model, file)

        file.close()




import json
import pickle
import numpy as np
import csv

from sklearn.model_selection import GridSearchCV

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
    
    def prepare_data_for_cnn(self,path):
        """Loads training dataset from json file.
            :param data_path (str): Path to json file containing data
            :return X (ndarray): Inputs
            :return y (ndarray): Targets
        """

        with open(path, "r") as fp:
            data = json.load(fp)

        X = np.array(data["pitch"])
        y = np.array(data["labels"])
        z = np.array(data["mapping"])

        X = X[..., np.newaxis]
        
        return X, y, z
    
    def load_data_csv(data_path):

        with open(data_path, "r") as fp:
            reader = csv.reader(fp)

            # Skip the header row if it exists
            next(reader, None)
            
            data = {}
            data["mapping"] = []
            data["pitch"] = []
            data["labels"] = []
            data["order"] = []

            for row in reader:
                data["mapping"].append(row[0])
                data["order"].append(row[1])

                # Extract individual pitch values from the string, convert to floats
                pitch_values = [float(x) for x in row[2].strip("[]").split(", ")]
                data["pitch"].append(pitch_values)
                data["labels"].append(int(row[3]))

        print(np.array(data["pitch"]))
        return data
    
    def train(model, epochs, batch_size, X_train, y_train):
        """Trains model
        :param epochs (int): Num training epochs
        :param batch_size (int): Samples per batch
        :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
        :param X_train (ndarray): Inputs for the train set
        :param y_train (ndarray): Targets for the train set
        :param X_validation (ndarray): Inputs for the validation set
        :param y_validation (ndarray): Targets for the validation set

        :return history: Training history
        """
        
        # train model
        history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size)
        
        return history


    def save_model(self,model,path):

        with open(path, 'wb') as file:  
            pickle.dump(model, file)

        file.close()

    def predict_chord(self,model, X_test,y_test, z,z_test):

        flg=0

        y_pred = model.predict(X_test)
        print("\nKNN:")
        
        for i in range(len(X_test)):
            print(z[y_pred[i]],end=' ' )
            if z[y_pred[i]] == z_test[i]:
                    flg+=1
        print(f"\nAccuracy = {flg}/{len(y_test)}")

    def GridSearcher(self,X_train,y_train,model,param_grid):
         
        
         
        grid_search = GridSearchCV(
            model, 
            param_grid=param_grid
        ) 
        
        grid_search.fit(X_train, y_train) 

        print(grid_search.best_estimator_) 
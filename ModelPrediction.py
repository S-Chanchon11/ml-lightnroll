import csv
from datetime import datetime
import logging as logger
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from Constant import TEST_DATA, TRAIN_DATA
from ModelMaster import ModelMaster
from Utilities import Utilities

logger.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logger.DEBUG,
    # filename="predicted_result_split_001.log",
    # filemode="a",
)


def init_data():

    utils = Utilities()

    X_train, y_train, z_train = utils.prepare_data(path=TRAIN_DATA)
    X_test, y_test, z_test = utils.prepare_data(path=TEST_DATA)

    MM = ModelMaster(
        X_train=X_train,
        y_train=y_train,
        z_train=z_train,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test
        )

    return MM

def init_split_data():

    utils = Utilities()

    X, y, z = utils.prepare_data(path=TRAIN_DATA)

    _X_test, _y_test, z_test = utils.prepare_data(path=TEST_DATA)

    X_train, X_test, y_train, y_test = utils.split_data(X=X,y=y)


    # logger.debug(z_test[y_test])
    
    MM = ModelMaster(
        X_train=X_train,
        y_train=y_train,
        z_train=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test
        )

    return MM


def fine_tuner(MM:ModelMaster):

    rf_model = RandomForestClassifier()
    # ann_model = MLPClassifier()
     
    # RandomForest
    param = {
            'n_estimators':[70,80,85,90,95,100,110,120], 
            'max_depth':[2,4,6,8], 
            'max_features':['log2',"sqrt"], 
            'max_leaf_nodes':[2,4,6,8]
        } 

    # param = {
    #     'hidden_layer_sizes':[(100, 50, 100),(50,100,150),(100,100,50),(150,100,50),(100,150,100)],
    #     'activation':["relu","tanh"],
    #     'solver':["adam"],
    #     'alpha':[0.05,0.001,0.005,0.0001],
    #     'max_iter':[300,280,250,200],
    #     'learning_rate':['constant', 'adaptive', 'invscaling']
    # }

    best_param = MM.fine_tuning(
        model=rf_model,
        param=param
    )

    return best_param


def prediction(MM:ModelMaster):
    
    # knn1 = MM.knn(n_neighbors=3)
    
    # rf1 = MM.rf(n_estimators=100, max_depth=6, max_features="log2", max_leaf_nodes=9)

    # rf2 = MM.rf(n_estimators=80, max_depth=4, max_features="log2", max_leaf_nodes=9)

    # rf3 = MM.rf(n_estimators=85, max_depth=4, max_features="log2", max_leaf_nodes=6)

    # rf4 = MM.rf(max_depth=4, max_features='log2', max_leaf_nodes=8,n_estimators=80)

    # rf4 = MM.rf(n_estimators=90, max_depth=6, max_features="log2", max_leaf_nodes=6)

    # ann1 = MM.ann(
    #     hidden_layer_sizes=(100, 50, 100),
    #     activation="relu",
    #     solver="adam",
    #     alpha=0.05,
    #     max_iter=300,
    #     learning_rate="constant",
    #     momentum=0.9,
    # )

    # ann2 = MM.ann(
    #     hidden_layer_sizes=(150, 200, 100),
    #     activation="relu",
    #     solver="adam",
    #     alpha=0.005,
    #     max_iter=300,
    #     learning_rate="constant",
    #     momentum=0.9,
    # )

    ann3 = MM.ann(
        hidden_layer_sizes=(200, 250, 150),
        activation="relu",
        solver="adam",
        alpha=0.05,
        max_iter=300,
        learning_rate="constant",
        momentum=0.9,
    )

    ann4 = MM.ann(
        hidden_layer_sizes=(150, 200, 100),
        activation="relu",
        solver="adam",
        alpha=0.05,
        max_iter=300,
        learning_rate="constant",
        momentum=0.9,
    )

    ann5 = MM.ann(
        hidden_layer_sizes=(150, 200, 100),
        activation="relu",
        solver="adam",
        alpha=0.05,
        max_iter=400,
        learning_rate="constant",
        momentum=0.9,
    )


    return ann3,ann4,ann5

    """
    0 : result
    1 : accuracy
    2 : param
    """
    

    # logger.debug("KNN   : %d/30 %s", (knn1[1],knn1[2]))
    # logger.debug("RF 1  : %d/30 %s", rf1[1],rf1[2])
    # logger.debug("RF 2  : %d/30 %s", rf2[1],rf2[2])
    # logger.debug("RF 3  : %d/30 %s", rf3[1],rf3[2])
    # logger.debug("ANN 1 : %d/30 %s", ann1[1],ann1[2])
    # logger.debug("ANN 2 : %d/30 %s", ann2[1],ann2[2])
    # logger.debug("ANN 3 : %d/30 %s", ann3[1],ann3[2])
    # logger.debug("ANN 4 : %d/30 %s", ann4[1],ann4[2])
    # logger.debug("ANN 5 : %d/30 %s", ann5[1],ann5[2])

    # logger.debug("RF 1  : %d/32", rf1[1])
    # logger.debug("RF 2  : %d/32", rf2[1])
    # logger.debug("RF 3  : %d/32", rf3[1])
    # logger.debug("RF 4  : %d/32", rf4[1])
    # logger.debug("ANN 1 : %d/32", ann1[1])
    # logger.debug("ANN 2 : %d/32", ann2[1])
    # logger.debug("ANN 3 : %d/32", ann3[1])
    # logger.debug("ANN 4 : %d/32", ann4[1])
    # logger.debug("ANN 5 : %d/32", ann5[1])
    # logger.debug("ANN 6 : %d/32", ann6[1])
    
    

    # logger.debug("KNN   : %s", (knn1[0]))
    # logger.debug("RF 1  : %s", rf1[0])
    # logger.debug("RF 2  : %s", rf2[0])
    # logger.debug("RF 3  : %s", rf3[0])
    # logger.debug("ANN 1 : %s", ann1[0])
    # logger.debug("ANN 2 : %s", ann2[0])
    # logger.debug("ANN 3 : %s", ann3[0])
    # logger.debug("ANN 4 : %s", ann4[0])
    # logger.debug("ANN 5 : %s", ann5[0])
    # logger.debug("ANN 6 : %s", ann6[0])
    
    
    

def main():

    # data = init_data()

    data = init_split_data()


    # result = fine_tuner(data)
    # print(f"Best param : {result}")

    prediction(data)

    # csv_path = "accuracy_improve_model_2.csv"
    # with open(csv_path, "a", newline="") as csv_file:
    #     writer = csv.writer(csv_file)

    #     if os.stat(csv_path).st_size == 0:
    #             writer.writerow(
    #                 [
    #                     "date-time",
    #                     "ANN3",
    #                     "ANN4",
    #                     "ANN5"
    #                 ]
    #             )

    #     for i in range(200): 
    #         data = init_split_data()   
    #         # print(".",end=' ')
    #         # time.sleep(3)
    #         # logger.debug(1)
    #         pred = prediction(data)

    #         dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #         # print("date and time =", dt_string)

    #         # write header row if the file doesn't exist
            

    #         # process all segments of audio file
    #         data = {
    #             "date-time":dt_string,
    #             "ANN3":pred[0][1],
    #             "ANN4":pred[1][1],
    #             "ANN5":pred[2][1]
    #         }
    #         print(data)
    #         # write data to the csv file
    #         writer.writerow(data.values())
    
if __name__ == "__main__":

    main()


    # print("\nCorrect order :\nG G G G G E E E E ")
    # print("\nCorrect order :\nAm Em G A F Dm C D E B ")
    # print("\nCorrect order :\nAm Am Am Em Em Em G G G A A A F F F Dm Dm Dm C C C D D D E E E B B B ")

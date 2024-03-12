import logging as logger
import time

from sklearn.ensemble import RandomForestClassifier
from Constant import TEST_DATA, TRAIN_DATA
from ModelMaster import ModelMaster
from Utilities import Utilities

logger.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logger.DEBUG,
    # filename="predicted_result_002.log",
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


def fine_tuner(MM:ModelMaster):

    rf_model = RandomForestClassifier()

    RandomForestParam = {
            'n_estimators':[80,85,90,95,100], 
            'max_depth':[2,4,6,8], 
            'max_features':['log2'], 
            'max_leaf_nodes':[2,4,6,8]
        }

    best_param = MM.fine_tuning(
        model=rf_model,
        param=RandomForestParam
    )

    return best_param


def prediction(MM:ModelMaster):
    

    knn1 = MM.knn(n_neighbors=3)

    rf1 = MM.rf(n_estimators=100, max_depth=6, max_features="log2", max_leaf_nodes=9)

    rf2 = MM.rf(n_estimators=80, max_depth=4, max_features="log2", max_leaf_nodes=9)

    rf3 = MM.rf(n_estimators=85, max_depth=4, max_features="log2", max_leaf_nodes=6)

    # rf4 = MM.rf(n_estimators=90, max_depth=6, max_features="log2", max_leaf_nodes=6)

    ann1 = MM.ann(
        hidden_layer_sizes=(100, 50, 100),
        activation="relu",
        solver="adam",
        alpha=0.05,
        max_iter=300,
        learning_rate="constant",
        momentum=0.9,
    )

    ann2 = MM.ann(
        hidden_layer_sizes=(100, 300, 100),
        activation="relu",
        solver="adam",
        alpha=0.05,
        max_iter=300,
        learning_rate="constant",
        momentum=0.9,
    )

    ann3 = MM.ann(
        hidden_layer_sizes=(100, 100, 50),
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
        hidden_layer_sizes=(100, 50, 100),
        activation="relu",
        solver="lbfgs",
        alpha=0.05,
        max_iter=300,
        learning_rate="constant",
        momentum=0.9,
    )

    """
    0 : result
    1 : accuracy
    2 : param
    """

    # logger.debug("KNN   : %d/30 %s", (knn1[1],knn1[2]))
    # logger.debug("RF 1  : %d/30 %s", rf1[1],rf1[2])
    # logger.debug("RF 2  : %d/30 %s", rf2[1],rf2[2])
    # logger.debug("RF 3  : %d/30 %s", rf3[1],rf3[2])
    # logger.debug("RF 4  : %d/30 %s", rf4[1],rf4[2])
    # logger.debug("ANN 1 : %d/30 %s", ann1[1],ann1[2])
    # logger.debug("ANN 2 : %d/30 %s", ann2[1],ann2[2])
    # logger.debug("ANN 3 : %d/30 %s", ann3[1],ann3[2])
    # logger.debug("ANN 4 : %d/30 %s", ann4[1],ann4[2])
    # logger.debug("ANN 5 : %d/30 %s", ann5[1],ann5[2])
    
    logger.debug("KNN   : %d/30", knn1[1])
    logger.debug("RF 1  : %d/30", rf1[1])
    logger.debug("RF 2  : %d/30", rf2[1])
    logger.debug("RF 3  : %d/30", rf3[1])
    # logger.debug("RF 4  : %d/30", rf4[1])
    logger.debug("ANN 1 : %d/30", ann1[1])
    logger.debug("ANN 2 : %d/30", ann2[1])
    logger.debug("ANN 3 : %d/30", ann3[1])
    logger.debug("ANN 4 : %d/30", ann4[1])
    logger.debug("ANN 5 : %d/30", ann5[1])
    

def main():

    data = init_data()

    # result = fine_tuner(data)
    # logger.debug(f"Best param : {result}")

    prediction(data)

    # for i in range(30):
    #     print("running")
    #     print(".",end='')
    #     time.sleep(5)
    #     # logger.debug(1)
    #     prediction(data)
    
if __name__ == "__main__":

    main()


    # print("\nCorrect order :\nG G G G G E E E E ")
    # print("\nCorrect order :\nAm Em G A F Dm C D E B ")
    # print("\nCorrect order :\nAm Am Am Em Em Em G G G A A A F F F Dm Dm Dm C C C D D D E E E B B B ")

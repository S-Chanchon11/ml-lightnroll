import logging as logger
import os
from Constant import TEST_DATA, TRAIN_DATA
from ModelMaster import ModelMaster
from Utilities import Utilities

logger.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logger.DEBUG,
    filename="predicted_result.log",
    filemode="a",
)


def main():

    utils = Utilities()

    X_train, y_train, z_train = utils.prepare_data(path=TRAIN_DATA)
    X_test, y_test, z_test = utils.prepare_data(path=TEST_DATA)

    mm = ModelMaster(
        X_train=X_train,
        y_train=y_train,
        z_train=z_train,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
    )

    knn = mm.knn(n_neighbors=5)

    rf = mm.rf(n_estimators=100, max_depth=6, max_features="log2", max_leaf_nodes=9)

    ann = mm.ann(
        hidden_layer_sizes=(100, 300, 100),
        activation="relu",
        solver="adam",
        alpha=0.05,
        max_iter=300,
    )

    logger.debug("KNN : %s", knn)
    logger.debug("RF : %s", rf)
    logger.debug("ANN : %s", ann)

    # print(f"\n{knn}\
    #       \n{rf}\
    #       \n{ann}\
    #         ")


if __name__ == "__main__":

    main()

    # print("\nCorrect order :\nG G G G G E E E E ")
    # print("\nCorrect order :\nAm Em G A F Dm C D E B ")
    # print("\nCorrect order :\nAm Am Am Em Em Em G G G A A A F F F Dm Dm Dm C C C D D D E E E B B B ")

import pickle
from sklearn.model_selection import train_test_split
from Constant import TEST_DATA, TRAIN_DATA
from ModelMaster import KNN, SVM, ModelMaster, RandomForest, CNN, ANN
from Utilities import Utilities


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
        z_test=z_test
        )
    
    knn = mm.knn(n_neighbors=5)

    rf = mm.rf(
        n_estimators=100,
        max_depth=6,
        max_features="log2",
        max_leaf_nodes=9
    )

    ann = mm.ann(
        hidden_layer_sizes=(100, 300, 100),
        activation="relu",
        solver="adam",
        alpha=0.05,
        max_iter=300,
    )

        
if __name__ == '__main__':

    main()

# utils = Utilities()
# cnn = CNN()
# knn = KNN()
# svm = SVM()
# rf = RandomForest()
# ann = ANN()


# def model_center(X_train, y_train, z, X_test, y_test, z_test):
#     print("\ntest size: ", len(y_test))
#     print("train size: ", len(y_train))

#     # for i in range(len(X_test)):

#     knn_result = knn.KNN(
#         save_path="knn_model.h5",
#         X=X_train,
#         y=y_train,
#         z=z,
#         X_test=X_test,
#         y_test=y_test,
#         z_test=z_test,
#         n_neighbors=1,  # ลอง odd number
#     )

#     svm_l_result = svm.SVM_linear(
#         X=X_train,
#         y=y_train,
#         z=z,
#         X_test=X_test,
#         y_test=y_test,
#         z_test=z_test,
#         degree=2,
#     )

#     svm_r_result = svm.SVM_radial(
#         X=X_train,
#         y=y_train,
#         z=z,
#         X_test=X_test,
#         y_test=y_test,
#         z_test=z_test,
#         degree=3,
#     )

#     rf_result = rf.RandomForest(
#         X=X_train,
#         y=y_train,
#         z=z,
#         X_test=X_test,
#         y_test=y_test,
#         z_test=z_test,
#         max_depth=6,
#         max_features="log2",
#         max_leaf_nodes=9,
#         # tree_n (*), leaf_n, depth_n
#     )

#     """
#         tree : risk of overfit

#     """

#     ann_result = ann.ANN(
#         X=X_train,
#         y=y_train,
#         z=z,
#         X_test=X_test,
#         y_test=y_test,
#         z_test=z_test,
#         hidden_layer_sizes=(100, 300, 100),
#         activation="relu",
#         solver="adam",
#         alpha=0.05,
#         max_iter=300,
       
        
#     )
#     """
#         learning rate : less = slow = accurate, 
#         momentum rate : same as learning rate, alpha, 
#         max_iter : ดูที่loss, if consistent then stop
#     """

#     print(
#         f"KNN : {knn_result}\
#             \nSVM linear : {svm_l_result}\
#             \nSVM radial : {svm_r_result}\
#             \nRandomForest : {rf_result}\
#             \nANN : {ann_result}"
#     )

#     # knn_model = pickle.load(open(KNN_MODEL, 'rb'))

#     # utils.predict_chord(knn_model,X_test=X_test,y_test=y_test,z=z,z_test=z_test)


# if __name__ == "__main__":

#     DATA_PATH = "output/data_all_chord_2.json"

#     # X, y, z = utils.prepare_data(DATA_PATH)
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#     X_train, y_train, z = utils.prepare_data(DATA_PATH)

#     TEST_PATH = "output/test/test_all_3.json"
#     # TEST_PATH = "output/test/test_Am.json"

#     X_test, y_test, z_test = utils.prepare_data(TEST_PATH)

#     for i in range(1):
#         model_center(
#             X_train=X_train,
#             y_train=y_train,
#             z=z,
#             X_test=X_test,
#             y_test=y_test,
#             z_test=z_test,
#         )

    # print("\nCorrect order :\nG G G G G E E E E ")
    # print("\nCorrect order :\nAm Em G A F Dm C D E B ")
    # print("\nCorrect order :\nAm Am Am Em Em Em G G G A A A F F F Dm Dm Dm C C C D D D E E E B B B ")

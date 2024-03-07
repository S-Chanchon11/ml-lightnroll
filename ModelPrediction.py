import pickle
from sklearn.model_selection import train_test_split
from ModelMaster import KNN, SVM, RandomForest
from CNN import CNN
from Utilities import Utilities

def model_center(X_train,y_train,z,X_test,y_test):


    # knn.KNN(
    #     save_path="knn_model.h5",
    #     X=X_train,
    #     y=y_train,
    #     z=z,
    #     X_test=X_test,
    #     y_test=y_test,
    #     n_neighbors=5
    # )

    # X_train_cnn, y_train_cnn, z_train_cnn = utils.prepare_data_for_cnn(DATA_PATH)
    # cnn.CNN(
    #     save_path="cnn_model.h5",
    #     X=X_train_cnn,
    #     y=y_train_cnn,
    #     epochs=100,
    #     batch_size=32,
    #     patience=5,
    #     learning_rate=0.001
    # )

    KNN_MODEL = "knn_model.h5"
    CNN_MODEL = "cnn_model.h5"

    print("test size: ", len(y_test))
    print("train size: ",len(y_train))

    svm.SVM(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        degree=3
    )

    rf.RandomForest(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        max_depth=6, 
        max_features='log2', 
        max_leaf_nodes=9
    )

    knn_model = pickle.load(open(KNN_MODEL, 'rb'))
    # cnn_model = pickle.load(open(CNN_MODEL, 'rb'))

    utils.predict_chord(knn_model,X_test=X_test,z=z)

    # print("\nCorrect order :\nAm Em G A F Dm C D B")

    # X_test, y_test, z_test = utils.prepare_data_for_cnn(TEST_PATH)

    # cnn.predict_chord(cnn_model,X_test=X_test,z=z)


if __name__ == "__main__":

    utils = Utilities()
    cnn = CNN()
    knn = KNN()
    svm = SVM()
    rf = RandomForest()

    DATA_PATH = "output/data_all_chord.json"

    # X, y, z = utils.prepare_data(DATA_PATH)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    
    X_train, y_train, z = utils.prepare_data(DATA_PATH)

    TEST_PATH = "output/test/test_all_2.json"
    X_test, y_test, z_test = utils.prepare_data(TEST_PATH)

    model_center(
        X_train=X_train,
        y_train=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test
    )

    # rf.GridSearcher(X_train=X_train,y_train=y_train)
    


    

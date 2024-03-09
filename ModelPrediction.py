import pickle
from sklearn.model_selection import train_test_split
from ModelMaster import KNN, SVM, RandomForest, CNN, ANN
from Utilities import Utilities

utils = Utilities()
cnn = CNN()
knn = KNN()
svm = SVM()
rf = RandomForest()
ann = ANN()


def model_center(X_train,y_train,z,X_test,y_test,z_test):


    # knn.KNN(
    #     save_path="knn_model.h5",
    #     X=X_train,
    #     y=y_train,
    #     z=z,
    #     X_test=X_test,
    #     y_test=y_test,
    #     n_neighbors=5 #ลอง odd number
    # )

    # X_train_cnn, y_train_cnn, z_train_cnn = utils.prepare_data_for_cnn(DATA_PATH)
    # cnn.CNN(
    #     save_path="cnn_model.h5",
    #     X=X_train_cnn,
    #     y=y_train_cnn,
    #     epochs=100,
    #     batch_size=32,
    #     patience=5,
    #     learning_rate=0.0001
    # )

    KNN_MODEL = "knn_model.h5"

    print("test size: ", len(y_test))
    print("train size: ",len(y_train))

    svm.SVM_linear(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
        degree=3
    )

    svm.SVM_radial(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
        degree=3
    )

    rf.RandomForest(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
        max_depth=6, 
        max_features='log2', 
        max_leaf_nodes=9
    )

    ann.ANN(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
    )

    knn_model = pickle.load(open(KNN_MODEL, 'rb'))

    utils.predict_chord(knn_model,X_test=X_test,y_test=y_test,z=z,z_test=z_test)


    

if __name__ == "__main__":


    DATA_PATH = "output/data_all_chord.json"

    # X, y, z = utils.prepare_data(DATA_PATH)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    
    X_train, y_train, z = utils.prepare_data(DATA_PATH)

    TEST_PATH = "output/test/test_multi.json"
    # TEST_PATH = "output/test/test_Am.json"
    
    X_test, y_test, z_test = utils.prepare_data(TEST_PATH)
    
    model_center(
        X_train=X_train,
        y_train=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test
    )

    print("\nCorrect order :\nG G G G E E E E")


    


    

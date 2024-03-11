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
    print("\ntest size: ", len(y_test))
    print("train size: ",len(y_train))

    # for i in range(len(X_test)):

    knn_result = knn.KNN(
        save_path="knn_model.h5",
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
        n_neighbors=5,  #ลอง odd number
        
    )

    svm_l_result = svm.SVM_linear(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
        degree=3,
    )

    svm_r_result = svm.SVM_radial(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
        degree=3,
    )

    rf_result = rf.RandomForest(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
        max_depth=6, 
        max_features='log2', 
        max_leaf_nodes=9,
    )

    ann_result = ann.ANN(
        X=X_train,
        y=y_train,
        z=z,
        X_test=X_test,
        y_test=y_test,
        z_test=z_test,
    )

        # print(f"KNN : {knn_result}\
        #       \nSVM linear : {svm_l_result}\
        #       \nSVM radial : {svm_r_result}\
        #       \nRandomForest : {rf_result}\
        #       \nANN : {ann_result}"
        #       )

    # knn_model = pickle.load(open(KNN_MODEL, 'rb'))

    # utils.predict_chord(knn_model,X_test=X_test,y_test=y_test,z=z,z_test=z_test)


    

if __name__ == "__main__":


    DATA_PATH = "output/data_all_chord_2.json"

    # X, y, z = utils.prepare_data(DATA_PATH)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    
    X_train, y_train, z = utils.prepare_data(DATA_PATH)

    TEST_PATH = "output/test/test_all_3.json"
    # TEST_PATH = "output/test/test_Am.json"
    
    X_test, y_test, z_test = utils.prepare_data(TEST_PATH)
    print(y_test)
    for i in range(2):
        model_center(
            X_train=X_train,
            y_train=y_train,
            z=z,
            X_test=X_test,
            y_test=y_test,
            z_test=z_test
        )

    # print("\nCorrect order :\nG G G G G E E E E ")
    # print("\nCorrect order :\nAm Em G A F Dm C D E B ")
    # print("\nCorrect order :\nAm Am Am Em Em Em G G G A A A F F F Dm Dm Dm C C C D D D E E E B B B ")





    

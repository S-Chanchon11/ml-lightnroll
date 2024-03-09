from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import tensorflow as tf
import numpy as np
from Utilities import Utilities

DATA_PATH = "data_pcp.json"
utils = Utilities()

class KNN:

    def KNN(self,save_path,X,y,z,X_test,y_test,z_test,n_neighbors):

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        #Create a KNN Classifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
        model.fit(X, y)

        y_pred = model.predict(X_test)
        print("\nKNN:")
        for i in range(len(X_test)):
            if y_test[i]==z_test[i]:
                print(z[y_pred[i]],end=' ')

        print(accuracy_score(y_test, y_pred))

        utils.save_model(model=model,path=save_path)

class SVM:

    def SVM_linear(self,X,y,z,X_test,y_test,z_test,degree,save_path=''):

        flg=0

        svclassifier_lin = SVC(
            kernel='linear',
            degree=degree
            )
    
        # Train the model using the training sets
        svclassifier_lin.fit(X, y)

        y_pred_lin = svclassifier_lin.predict(X_test)
        print(len(z_test))
        print("\nSVM linear: ")
        for i in range(len(X_test)):
            print(z[y_pred_lin[i]],end=' ' )
            if z[y_pred_lin[i]] == z_test[i]:
                flg+=1
        print(f"\nAccuracy = {flg}/{len(y_test)}")
        # print("\nAccuracy : ",accuracy_score(y_test, y_pred_lin)*100)

            
    def SVM_radial(self,X,y,z,X_test,y_test,z_test,degree,save_path=''):

        flg=0

        svclassifier_rbf = SVC(
            kernel='rbf',
            degree=degree
            )
    
        svclassifier_rbf.fit(X, y)

        y_pred_rbf = svclassifier_rbf.predict(X_test)

        print("\nSVM rbf: ")
        for i in range(len(X_test)):
            print(z[y_pred_rbf[i]],end=' ' )
            if z[y_pred_rbf[i]] == z_test[i]:
                flg+=1
        print(f"\nAccuracy = {flg}/{len(y_test)}")
        # print("\nAccuracy : ",accuracy_score(y_test, y_pred_rbf)*100)
        
        # utils.save_model(model=svclassifier_lin,path=save_path)

class RandomForest:

    def RandomForest(self,X,y,z,X_test,y_test,z_test,max_depth, max_features, max_leaf_nodes):

        flg=0

        # RandomForest_Param = {
        #     'hidden_layer_sizes': [(10,30,10),(20,)],
        #     'activation': ['tanh', 'relu'],
        #     'solver': ['sgd', 'adam'],
        #     'alpha': [0.0001, 0.05],
        #     'learning_rate': ['constant','adaptive'],
        # }

        rf = RandomForestClassifier(
            max_depth=max_depth, 
            max_features=max_features, 
            max_leaf_nodes=max_leaf_nodes
            )

        rf.fit(
            X, 
            y
            )

        y_pred = rf.predict(X_test)

        print("\nRandomForest:")
        for i in range(len(X_test)):
            print(z[y_pred[i]],end=' ')

            if z[y_pred[i]] == z_test[i]:
                flg+=1

        print(f"\nAccuracy = {flg}/{len(y_test)}")

    

class CNN:
    
    def build_cnn(self,input_shape,learning_rate):
    
        model = tf.keras.models.Sequential()

        # 1st conv layer
        model.add(tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3), 
            activation='relu', 
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            padding='same'
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), 
            strides=(2,2), 
            padding='same'))

        # 2nd conv layer
        model.add(tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3), 
            activation='relu', 
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            padding='same'
            )
        )

        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), 
            strides=(2,2), 
            padding='same'))

        # 3rd conv layer
        model.add(tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(2, 2), 
            activation='relu', 
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            padding='same'
            )
        )

        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), 
            strides=(2,2), 
            padding='same'))

        # flatten output and feed into dense layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(80, activation='relu'))
        tf.keras.layers.Dropout(0.3)

        # softmax output layer
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        loss="sparse_categorical_crossentropy"

        optimiser = tf.optimizers.legacy.Adam(learning_rate=learning_rate)

        # compile model
        model.compile(optimizer=optimiser,
                    loss=loss,
                    metrics=["accuracy"]
                    )

        model.summary()

        return model
    
    
    def CNN(self,save_path,X,y,epochs,batch_size,patience,learning_rate,validation_size=0):

        train_shape = (X.shape[1], X.shape[2], 1)
        # X_temp, X_validation, y_temp, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

        
        # predict_model()
        model = self.build_cnn(input_shape=train_shape,learning_rate=learning_rate)

        model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            )

        utils.save_model(model=model,path=save_path)



    def predict_chord(self,model, X_test,z):
    
        _mapping = [
            "Am",
            "Em",
            "G",
            "A",
            "F",
            "Dm",
            "C",
            "D",
            "E",
            "B"
        ]
        
        print("\nCNN:")
        
        # for i in range(len(X_test)):
        predictions = model.predict(X_test)
        predicted_index = np.argmax(predictions)
        predicted_chord = _mapping[predicted_index]
        print(predicted_chord)

class ANN:

    def ANN(self,X,y,z,X_test,y_test,z_test):

        flg=0

        # ANN_Param = {
        #     'hidden_layer_sizes': [(10,30,10),(20,)],
        #     'activation': ['tanh', 'relu'],
        #     'solver': ['sgd', 'adam'],
        #     'alpha': [0.0001, 0.05],
        #     'learning_rate': ['constant','adaptive'],
        # }

        model = MLPClassifier(
            activation='relu', 
            alpha=0.05, 
            hidden_layer_sizes=(100, 300, 100),
            solver='adam',
            max_iter=300
        ) 

        model.fit(
            X,
            y
        )

        # utils.GridSearcher(
        #     X_train=X,
        #     y_train=y,
        #     model=model,
        #     param_grid=ANN_Param
        #     )

        y_pred = model.predict(X_test)
        print("\nANN")
        for i in range(len(X_test)):
            print(z[y_pred[i]],end=' ' )
            if z[y_pred[i]] == z_test[i]:
                flg+=1
        print(f"\nAccuracy = {flg}/{len(y_test)}")


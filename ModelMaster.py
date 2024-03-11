from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import _testing
from sklearn.exceptions import ConvergenceWarning


class ModelMaster:

    def __init__(self, X_train, y_train, z_train, X_test, y_test, z_test):
        self.X_train = X_train
        self.y_train = y_train
        self.z_train = z_train
        self.X_test = X_test
        self.y_test = y_test
        self.z_test = z_test

    def predict(self, model):

        flg = 0
        chord_pred = []

        y_pred = model.predict(self.X_test)

        for i in range(len(self.X_test)):

            chord_pred.append(self.z_train[y_pred[i]])

            if self.z_train[y_pred[i]] == self.z_test[self.y_test[i]]:

                flg += 1

        print(f"\nAccuracy = {flg}/{len(self.y_test)}")

        return chord_pred

    def train(self,model):

        model.fit(self.X_train, self.y_train)

        return model
    
    def knn(self,n_neighbors):

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors
        )

        trained_knn = self.train(knn)

        pred_knn = self.predict(trained_knn)

        return pred_knn
    
    def rf(self,n_estimators,max_depth, max_features, max_leaf_nodes):

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
        )

        trained_rf = self.train(rf)

        pred_rf = self.predict(trained_rf)

        return pred_rf
    
    @_testing.ignore_warnings(category=ConvergenceWarning)
    def ann(self,hidden_layer_sizes,activation,solver,alpha,max_iter):

        ann = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
        )

        trained_ann = self.train(ann)

        pred_ann = self.predict(trained_ann)

        return pred_ann






        



# class KNN:

#     def KNN(self, save_path, X, y, z, X_test, y_test, z_test, n_neighbors):

#         flg = 0
#         chord_pred = []

#         # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#         # Create a KNN Classifier
#         model = KNeighborsClassifier(n_neighbors=n_neighbors)

#         model.fit(X, y)

#         y_pred = model.predict(X_test)
#         # print("\nKNN:")
#         # return z[y_pred[i]]
#         for i in range(len(X_test)):
#             # if y_test[i]==z_test[i]:
#             # print(z[y_pred[i]],end=' ')
#             chord_pred.append(z[y_pred[i]])
#             # print(z_test.shape)
#             if z[y_pred[i]] == z_test[y_test[i]]:
#                 flg += 1

#         print(f"\nAccuracy = {flg}/{len(y_test)}")

#         return chord_pred
#         # print(accuracy_score(y_test, y_pred))
#         # utils.save_model(model=model,path=save_path)


# class SVM:

#     def SVM_linear(self, X, y, z, X_test, y_test, z_test, degree, save_path=""):

#         flg = 0
#         chord_pred = []
#         svclassifier_lin = SVC(kernel="poly", degree=degree)

#         # Train the model using the training sets
#         svclassifier_lin.fit(X, y)

#         y_pred_lin = svclassifier_lin.predict(X_test)

#         # print("\nSVM linear: ")
#         # return z[y_pred_lin[i]]
#         for i in range(len(X_test)):
#             # print(z[y_pred_lin[i]],end=' ' )
#             chord_pred.append(z[y_pred_lin[i]])
#             if z[y_pred_lin[i]] == z_test[y_test[i]]:
#                 flg += 1
#         print(f"\nAccuracy = {flg}/{len(y_test)}")

#         return chord_pred
#         # print("\nAccuracy : ",accuracy_score(y_test, y_pred_lin)*100)

#     def SVM_radial(self, X, y, z, X_test, y_test, z_test, degree, save_path=""):

#         flg = 0
#         chord_pred = []
#         svclassifier_rbf = SVC(kernel="rbf", degree=degree)

#         svclassifier_rbf.fit(X, y)

#         y_pred_rbf = svclassifier_rbf.predict(X_test)

#         # print("\nSVM rbf: ")
#         # return z[y_pred_rbf[i]]
#         for i in range(len(X_test)):
#             # print(z[y_pred_rbf[i]],end=' ' )
#             chord_pred.append(z[y_pred_rbf[i]])
#             if z[y_pred_rbf[i]] == z_test[y_test[i]]:
#                 flg += 1
#         print(f"\nAccuracy = {flg}/{len(y_test)}")

#         return chord_pred
#         # print("\nAccuracy : ",accuracy_score(y_test, y_pred_rbf)*100)

#         # utils.save_model(model=svclassifier_lin,path=save_path)


# class RandomForest:

#     def RandomForest(
#         self, X, y, z, X_test, y_test, z_test, max_depth, max_features, max_leaf_nodes
#     ):

#         flg = 0
#         chord_pred = []
#         RandomForest_Param = {"max_depth": [3, 4, 5], "max_leaf_nodes": [3, 4, 5]}

#         rf = RandomForestClassifier(
#             max_depth=max_depth,
#             max_features=max_features,
#             max_leaf_nodes=max_leaf_nodes,
#         )

#         utils.GridSearcher(
#             X_train=X, y_train=y, model=rf, param_grid=RandomForest_Param
#         )

#         rf.fit(X, y)

#         y_pred = rf.predict(X_test)
#         # print("\nRandomForest:")
#         # return z[y_pred[i]]

#         for i in range(len(X_test)):
#             # print(z[y_pred[i]],end=' ')
#             chord_pred.append(z[y_pred[i]])
#             if z[y_pred[i]] == z_test[y_test[i]]:
#                 flg += 1

#         print(f"\nAccuracy = {flg}/{len(y_test)}")

#         return chord_pred


# class ANN:

#     @_testing.ignore_warnings(category=ConvergenceWarning)
#     def ANN(
#         self,
#         X,
#         y,
#         z,
#         X_test,
#         y_test,
#         z_test,
#         hidden_layer_sizes,
#         activation,
#         solver,
#         alpha,
#         max_iter,
#     ):

#         flg = 0
#         chord_pred = []
#         ANN_Param = {
#             "hidden_layer_sizes": [(10, 30, 10), (20,)],
#             "activation": ["tanh", "relu"],
#             "solver": ["sgd", "adam"],
#             "alpha": [0.0001, 0.05],
#             "learning_rate": ["constant", "adaptive"],
#         }

#         model = MLPClassifier(
#             hidden_layer_sizes=hidden_layer_sizes,
#             activation=activation,
#             solver=solver,
#             alpha=alpha,
#             max_iter=max_iter,
#         )

#         model.fit(X, y)

#         utils.GridSearcher(X_train=X, y_train=y, model=model, param_grid=ANN_Param)

#         y_pred = model.predict(X_test)
#         # print("\nANN")
#         # return z[y_pred[i]]
#         for i in range(len(X_test)):
#             # print(z[y_pred[i]],end=' ' )
#             chord_pred.append(z[y_pred[i]])
#             if z[y_pred[i]] == z_test[y_test[i]]:
#                 flg += 1
#         print(f"\nAccuracy = {flg}/{len(y_test)}")

#         return chord_pred

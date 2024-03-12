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

        return chord_pred, flg

    def train(self,model):

        model.fit(self.X_train, self.y_train)

        return model

    def fine_tuning(self,model,param):
    
        grid_search = GridSearchCV(model, param)

        grid_search.fit(self.X_train, self.y_train)

        return grid_search.best_estimator_

    def knn(self,n_neighbors):

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors
        )

        trained_knn = self.train(knn)

        param = trained_knn.get_params()

        pred_knn, acc = self.predict(trained_knn)

        return pred_knn, acc, param
    
    def rf(self,n_estimators,max_depth, max_features, max_leaf_nodes):

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
        )

        trained_rf = self.train(rf)

        param = trained_rf.get_params()

        pred_rf, acc = self.predict(trained_rf)

        return pred_rf, acc, param
    

    @_testing.ignore_warnings(category=ConvergenceWarning)
    def ann(self,hidden_layer_sizes,activation,solver,alpha,max_iter,learning_rate,momentum):

        """
        learning rate : less = slow = accurate, 
        momentum rate : same as learning rate, alpha, 
        max_iter : ดูที่loss, if consistent then stop

        """

        ann = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            learning_rate=learning_rate,
            momentum=momentum
        )

        trained_ann = self.train(ann)

        param = trained_ann.get_params()

        pred_ann, acc = self.predict(trained_ann)

        return pred_ann, acc, param


class MagicParam:

    def __init__(self):
        self.RandomForestParam = {
            'n_estimators':[80], 
            'max_depth':[4], 
            'max_features':['log2'], 
            'max_leaf_nodes':[6]
        }
        self.ANNParam = {
            'hidden_layer_sizes':[(100, 300, 100)],
            'activation':["relu"],
            'solver':["adam"],
            'alpha':[0.05],
            'max_iter':[300],
            'learning_rate':["constant"],
            'momentum':[0.9]
        }
    
        




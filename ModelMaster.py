from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import _testing
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

class ModelMaster:

    def __init__(self, X_train, y_train, z_train, X_test, y_test, z_test):
        self.X_train = X_train
        self.y_train = y_train
        self.z_train = z_train
        self.X_test = X_test
        self.y_test = y_test
        self.z_test = z_test

    def predict(self, model):

        mp = MagicPlot()

        flg = 0
        chord_pred = []

        y_pred = model.predict(self.X_test)

        cm = confusion_matrix(self.y_test,y_pred)

        mp.plot_cm()

        for i in range(len(self.X_test)):

            chord_pred.append(self.z_train[y_pred[i]])

            if self.z_train[y_pred[i]] == self.z_test[self.y_test[i]]:

                flg += 1

        return chord_pred, flg

    def train(self,model):

        model.fit(self.X_train, self.y_train)

        return model

    def fine_tuning(self,model,param):
    
        grid_search = GridSearchCV(model, param, scoring='accuracy')

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


class MagicPlot:

    def plot_cm(cm):
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
    
        




import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import _testing
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

class ModelMaster:

    def __init__(self, X,y,X_train, y_train, z_train, X_test, y_test, z_test):
        self.X = X
        self.y = y
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
        mp = MagicPlot(self.y_test,y_pred)
        # # print(classification_report(self.y_test,y_pred))
        cm = confusion_matrix(self.y_test,y_pred)
        mp.plot_cm()

        
        # mp.cr_report()

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
    def ann(self,hidden_layer_sizes,activation,solver,alpha,max_iter,learning_rate,momentum,name=''):

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

        # scores = cross_val_score(ann, self.X, self.y, cv=5)
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        trained_ann = self.train(ann)

        # self.save_model(trained_ann,name)

        param = trained_ann.get_params()

        pred_ann, acc = self.predict(trained_ann)

        return pred_ann, acc, param
    
    def save_model(self, model, name):

        pickle.dump(model, open(name, 'wb'))

        # with open(path, "wb") as file:
            
        #     pickle.dumps(model, file)

        # file.close()


class MagicPlot:

    def __init__(self,y_test,y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_cm(self):

        cm = confusion_matrix(self.y_test,self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
    
    def cr_report(self):

        report = classification_report(self.y_test,self.y_pred)
        print(report)
        




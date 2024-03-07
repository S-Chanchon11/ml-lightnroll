from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from Utilities import Utilities

DATA_PATH = "data_pcp.json"
utils = Utilities()

class KNN:

    def KNN(self,save_path,X,y,z,X_test,y_test,n_neighbors):

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        #Create a KNN Classifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
        model.fit(X, y)

        y_pred = model.predict(X_test)
        print("\nKNN:")
        for i in range(len(X_test)):
            if y_test[i]!=y_pred[i]:
                print(z[y_pred[i]],end=' ')

        print(accuracy_score(y_test, y_pred))

        utils.save_model(model=model,path=save_path)

class SVM:

    def SVM(self,X,y,z,X_test,y_test,degree,save_path=''):

        svclassifier_lin = SVC(
            kernel='linear',
            degree=degree
            )
        svclassifier_rbf = SVC(
            kernel='rbf',
            degree=degree
            )
    
        # Train the model using the training sets
        svclassifier_lin.fit(X, y)
        svclassifier_rbf.fit(X, y)

        y_pred_lin = svclassifier_lin.predict(X_test)
        y_pred_rbf = svclassifier_rbf.predict(X_test)

        # tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lin).ravel()

        print("\nSVM linear: ")
        for i in range(len(X_test)):
            print(z[y_pred_lin[i]],end=' ' )
        # print("\nAccuracy : ",accuracy_score(y_test, y_pred_lin)*100)

        print("\nSVM rbf: ")
        for i in range(len(X_test)):
            print(z[y_pred_rbf[i]],end=' ' )
        # print("\nAccuracy : ",accuracy_score(y_test, y_pred_rbf)*100)
        
        # utils.save_model(model=svclassifier_lin,path=save_path)

class RandomForest:

    def RandomForest(self,X,y,z,X_test,y_test,max_depth, max_features, max_leaf_nodes):

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
            if y_test[i]!=y_pred[i]:
                print(z[y_pred[i]],end=' ')

    def GridSearcher(self,X_train,y_train):
         
        param_grid = { 
            'n_estimators': [25, 50, 100, 150], 
            'max_features': ['sqrt', 'log2', None], 
            'max_depth': [3, 6, 9], 
            'max_leaf_nodes': [3, 6, 9], 
        } 
         
        grid_search = GridSearchCV(
            RandomForestClassifier(), 
            param_grid=param_grid
        ) 
        
        grid_search.fit(X_train, y_train) 

        print(grid_search.best_estimator_) 




import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import pickle


rounds = 1000
Acc_lin_avg = 0
Acc_rbf_avg = 0
err_count_lin = 0
err_count_rbf = 0

# LOOP:
for i in range(rounds):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Create a SVM Classifier
    svclassifier_lin = SVC(kernel="linear")
    svclassifier_rbf = SVC(kernel="rbf")

    # Train the model using the training sets
    svclassifier_lin.fit(X_train, y_train)
    svclassifier_rbf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred_lin = svclassifier_lin.predict(X_test)
    y_pred_rbf = svclassifier_rbf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    Acc_lin_avg += metrics.accuracy_score(y_test, y_pred_lin)
    Acc_rbf_avg += metrics.accuracy_score(y_test, y_pred_rbf)

    for i in range(len(y_test)):
        if y_test[i] != y_pred_rbf[i]:
            err_count_rbf += 1
            # print("error in:",i," test: ",y_test[i], "pred: ", y_pred[i])
        if y_test[i] != y_pred_lin[i]:
            err_count_lin += 1

print("~~~Errors Comparison: ~~~")
print(
    "Type of SVM: Linear , number of errors: ",
    err_count_lin / rounds,
    "accuracy: ",
    Acc_lin_avg / rounds,
)
print(
    "Type of SVM: radial basis function , number of errors: ",
    err_count_rbf / rounds,
    "accuracy: ",
    Acc_rbf_avg / rounds,
)

pickle.dump(svclassifier_rbf, open("model.sav", "wb"))

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import time
from sklearn.metrics import accuracy_score,f1_score, classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# this module is basically for training the model
# you can change whichever model you want for the majority voting 

def model_train():
    print("model training....")
    with open('train_test_split.pkl', 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # here delete or add more models as per your wish
    clf1 = LogisticRegression(max_iter=1000)
    clf2 = KNeighborsClassifier()
    clf3 = DecisionTreeClassifier()
    clf4 = RandomForestClassifier(n_estimators=100)
    clf5 = SVC(kernel='linear', probability=True) # make sure here probability is always true
    clf6 = GaussianNB()
    


    voting_clf = VotingClassifier(
        estimators=[
            ('lr', clf1),
            ('knn', clf2),
            ('dt', clf3),
            ('rf', clf4),
            ('svm', clf5),
            ('nb', clf6),
        ],
        voting='soft')  # using soft is important

    selector=SelectKBest(mutual_info_classif,k=100)
    selector.fit(X_train_scaled,y_train)
    x_train_new=selector.transform(X_train_scaled)
    x_test_new=selector.transform(X_test_scaled)

    start = time.time()
    voting_clf.fit(x_train_new, y_train)
    voting_time = time.time() - start

    y_pred_voting = voting_clf.predict(x_test_new)
    voting_accuracy = round(accuracy_score(y_test, y_pred_voting),3)
    print("Voting Classifier - Train Time:", voting_time)
    print("Voting Classifier - Accuracy:", voting_accuracy)
    print(classification_report(y_test, y_pred_voting))
    
    return voting_clf, selector, scaler

if __name__ == "__main__":
    
    voting_clf, selector, scaler = model_train()
    
    with open('voting_classifier.pkl', 'wb') as f:
        pickle.dump(voting_clf, f)
    with open('selector.pkl', 'wb') as f:
        pickle.dump(selector, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[INFO] model training done!!")
  
  
    
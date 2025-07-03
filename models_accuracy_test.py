import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest


# the purpose of this module was to test the different models 
# and select the number of features to be used for the final model

if __name__ == "__main__":
  with open('train_test_split.pkl', 'rb') as f:
        data = pickle.load(f)

  X_train = data['X_train']
  X_test = data['X_test']
  y_train = data['y_train']
  y_test = data['y_test']
  print("models testing accuracy....")
  models = {
  'LogisticRegression': LogisticRegression(max_iter=1000),
  'KNN': KNeighborsClassifier(n_neighbors=3),
  'DecisionTree': DecisionTreeClassifier(),
  'RandomForest': RandomForestClassifier(n_estimators=100),
  'SVM': SVC(),
  'NaiveBayes': GaussianNB(),
  }


  results = []

  for name, model in models.items():
    max_accuracy = float('-inf')
    max_train_time = float('-inf')
    features_used=float('-inf')
    for i in [10,40,50,70,100,128]:
      start = time.time()
      selector=SelectKBest(mutual_info_classif,k=i)
      selector.fit(X_train,y_train)
      x_train_new=selector.transform(X_train)
      x_test_new=selector.transform(X_test)
      model.fit(x_train_new, y_train)
      train_time = time.time() - start


      y_pred = model.predict(x_test_new)
      accuracy = round(accuracy_score(y_test, y_pred),3)
      if accuracy>max_accuracy:
        max_accuracy=accuracy
        max_train_time=train_time
        features_used=i
        
    print("for "+name+" : done ")
    results.append([name, max_train_time, max_accuracy, features_used])


  results_df = pd.DataFrame(results, columns=['Model', 'Train Time (s)', 'Accuracy', 'features_used'])
  print(results_df)
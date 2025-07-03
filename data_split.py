
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# this is for splitting the dataset to training and testing batches

def data_split():
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Extract X and y
    X = data['X']
    y = data['y']
    X=np.array(X)
    y=np.array(y)

    X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

    return X_train, X_test,y_train,y_test

if __name__ == "__main__":
    print("Running data split...")
    X_train, X_test,y_train,y_test=data_split()
    with open('train_test_split.pkl', 'wb') as f:
        pickle.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}, f)
    
    print(f"Created split: {len(X_train)} train, {len(X_test)} test")

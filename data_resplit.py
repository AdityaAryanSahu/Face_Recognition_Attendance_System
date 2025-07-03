from data_split import data_split
import pickle

def resplit_data():
    X_train, X_test,y_train,y_test=data_split()
    print(f"[INFO] Resplit done. Train: {len(X_train)}, Test: {len(X_test)}")
    with open('train_test_split.pkl', 'wb') as f:
        pickle.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}, f) 
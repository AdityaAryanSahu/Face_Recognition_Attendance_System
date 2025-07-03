from model_train import model_train
import pickle

# this is for retraining the model after new registrations
# just use this whenever you wanna retrain 

def retrain_model():
    print("model retraining....")
    voting_clf, selector, scaler = model_train()
    
    with open('voting_classifier.pkl', 'wb') as f:
        pickle.dump(voting_clf, f)
    with open('selector.pkl', 'wb') as f:
        pickle.dump(selector, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("[INFO] model retraining done!!")
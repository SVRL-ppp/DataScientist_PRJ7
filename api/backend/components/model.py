import pickle
import pandas as pd
import xgboost
import sklearn
import os

path = os.path.split(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])[0] + '/data/'
application_test = pd.read_csv(path + "application_test_df.csv")

with open(path + 'model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(path + 'feats.pkl', 'rb') as f:
    feats = pickle.load(f)

def model_prediction(SKID):
    yprob = model.predict_proba(application_test[feats].loc[application_test["SK_ID_CURR"]==int(SKID)])
    return yprob
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def get_model(X, y, model_name, scoring='accuracy'):
    gs_args = {
        'n_jobs': -1,
        'refit': True,
        'cv': 5,
        'scoring': scoring
    }

    model_dict = {
        "xgb": GridSearchCV(XGBClassifier(), 
            param_grid={'max_depth': list(range(1, 7))}, **gs_args),
        "rf": GridSearchCV(RandomForestClassifier(), 
            param_grid={'max_depth': list(range(1, 7))}, **gs_args)
    }

    clf = model_dict[model_name]

    if model_name in ['xgb', 'rf']:
        clf.fit(X, y)

    return clf

def train_clf(model_name, X_train, y_train, X_test, y_test, scoring='accuracy'):
    clf = get_model(X_train, y_train, model_name, scoring=scoring)
    
    predictions_test = clf.predict(X_test)
    proba_test = clf.predict_proba(X_test)[:, 1] 
    
    test_accuracy = accuracy_score(y_test, predictions_test)
    
    return clf, predictions_test, proba_test, test_accuracy

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds


def get_fair_model(X, y, sensitive_attr, model, reductionist_type, adjusted_bound):
    """Train a model with fairness constraints."""
    if reductionist_type == 'DP':
        constraint = DemographicParity(difference_bound=adjusted_bound)
    elif reductionist_type == 'EO':
        constraint = EqualizedOdds(difference_bound=adjusted_bound)
    else:
        raise ValueError("Unsupported reductionist type")
    
    eg_un = ExponentiatedGradient(model, constraint, eps = adjusted_bound)
    
    eg = ExponentiatedGradient(model, constraint, eps = adjusted_bound)
    eg.fit(X, y, sensitive_features=sensitive_attr)
    return eg, eg_un

def train_clf_fair(model_name, X_train, y_train, X_test, y_test, s_train, constraint, adjusted_bound = None):
    """Train and evaluate a fair classifier."""
  
    model_dict = {
        "xgb": GridSearchCV(XGBClassifier(), 
            param_grid={'max_depth': list(range(1, 7))}),
        "rf": GridSearchCV(RandomForestClassifier(),
            param_grid={'max_depth': list(range(1, 7))})
    }

    if model_name not in model_dict:
        raise ValueError("Model not supported")
    
    fair_model, model = get_fair_model(X_train, y_train, s_train, model_dict[model_name], constraint, adjusted_bound)

    # Make predictions and calculate accuracy
    predictions_test = fair_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions_test)

    return fair_model, model, predictions_test, test_accuracy
import fairlearn

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def demographic_parity_diff(true, pred, sensitive_attri):
    
    return demographic_parity_difference(y_true=true, y_pred = pred, sensitive_features=sensitive_attri, method='between_groups')

def equalized_odds_diff(true, pred, sensitive_attri):
    
    return equalized_odds_difference(y_true = true, y_pred = pred, sensitive_features=sensitive_attri,method='between_groups')
 
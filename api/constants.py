from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

class Constants():

    def __init__(self) -> None:
        pass


    def _get_classifiers(self): 
        return {
            'v1.0': {
                'lr': LogisticRegression(class_weight='balanced',random_state=0,max_iter=1000),
                'rf': RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=5, min_samples_leaf=5),
                'Xv': XGBClassifier(),
                'Sv': SVC(probability=True, kernel='linear', C=0.7, max_iter=1000)
            },
            'v2.0': {
                'lr': LogisticRegression(class_weight='balanced',random_state=0),
                'rf': RandomForestClassifier(),
                'Xv': XGBClassifier(),
                'Sv': SVC(probability=True, kernel='linear', C=0.7, max_iter=1000)
            }
        }
    

    def _get_configs_train(self):
        return {
            'v1.0': {
                'removing_cols_for_train': ['Date', 'Symbol', 'Dividends', 'Stock Splits'],  # Removing cols when training and predict
                'min_volume_prep_models': 250_000, # Define the minimum daily volume that must be considered when training
                'clean_targets_prep_models': True  # If True, remove outliers when training (beta)
            },
            'v2.0': {
                'removing_cols_for_train': ['Date', 'Dividends', 'Stock Splits'], # Removing cols when training and predict
                'min_volume_prep_models': 250_000, # Define the minimum daily volume that must be considered when training
                'clean_targets_prep_models': True  # If True, remove outliers when training (beta)
            }
        }
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
            },
            'v2.1.20': {
                'removing_cols_for_train': ['Date', 'Symbol', 'Dividends', 'Stock Splits'],  # Removing cols when training and predict
                'min_volume_prep_models': 250_000, # Define the minimum daily volume that must be considered when training
                'clean_targets_prep_models': True  # If True, remove outliers when training (beta)
            }
        }
    
    ### atualmente as colunas são retiradas diretamente
    def _remove_features(self):
        return {
            'v2.2.20': [
                '0', 'SUPERTs_200_3.0', 'HILOl_13_21', 'SUPERTl_50_3.0', 'SUPERTl_200_3.0', 'PSARl_0.02_0.2', 'SUPERTs_100_3.0', 'QQEs_14_5_4.236', 
                'SUPERTl_12_3.0', 'PSARs_0.02_0.2', 'SUPERTs_26_3.0', 'SUPERTl_26_3.0', 'QQEl_14_5_4.236', 'SUPERTl_5_3.0', 
                'SUPERTs_5_3.0', 'HILOs_13_21', 'SUPERTs_50_3.0', 'SUPERTl_100_3.0', 'SUPERTs_12_3.0'
            ]
        }
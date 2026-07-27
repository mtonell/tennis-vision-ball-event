from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

GLOBAL_KWARGS = {}

def get_ml_model(model_name, num_classes=2, **kwargs):
    if model_name == 'catboost':
        loss = 'MultiClass' if num_classes > 2 else 'Logloss'
        params = {'iterations': 200, 'depth': 6, 'learning_rate': 0.05, 'loss_function': loss, 'verbose': 0, 'auto_class_weights': 'Balanced'}
        params.update(kwargs)
        params.update(GLOBAL_KWARGS)
        return CatBoostClassifier(**params)
    elif model_name == 'rf':
        params = {'n_estimators': 200, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42}
        params.update(kwargs)
        params.update(GLOBAL_KWARGS)
        return RandomForestClassifier(**params)
    elif model_name == 'xgboost':
        objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'objective': objective, 'random_state': 42}
        params.update(kwargs)
        params.update(GLOBAL_KWARGS)
        return XGBClassifier(**params)
    elif model_name == 'lgbm':
        objective = 'multiclass' if num_classes > 2 else 'binary'
        params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'objective': objective, 'class_weight': 'balanced', 'random_state': 42, 'verbose': -1}
        params.update(kwargs)
        params.update(GLOBAL_KWARGS)
        return LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown ML model: {model_name}")

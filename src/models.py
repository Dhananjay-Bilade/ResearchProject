import numpy as np
import joblib
from xgboost import XGBClassifier

def train_xgboost(X_train, y_train, save_path='../models/xgboost_paysim.pkl'):
    ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=ratio,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model
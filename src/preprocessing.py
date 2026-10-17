import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data_loader import load_paysim_data

def preprocess_and_resample(file_path='../data/raw/PaySim_log.csv'):
    df = load_paysim_data(file_path)
    
    # Filter transfer & cash_out types
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    
    # Feature Engineering (Balance Errors)
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # Encoding & Dropping redundant columns
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True, errors='ignore')
    
    # Robust Scaling
    scaler = RobustScaler()
    cols_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Targeted SMOTE (10% ratio)
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, y_train_res, X_test, y_test
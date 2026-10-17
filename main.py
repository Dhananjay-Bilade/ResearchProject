from src.preprocessing import preprocess_and_resample
from src.models import train_xgboost
from src.evaluation import evaluate_model

if __name__ == "__main__":
    print("--- 1. Data Loading & Feature Engineering (PaySim) ---")
    X_train, y_train, X_test, y_test = preprocess_and_resample('data/raw/PaySim_log.csv')
    
    print("--- 2. Model Training (XGBoost) ---")
    model = train_xgboost(X_train, y_train, save_path='models/xgboost_paysim.pkl')
    
    print("--- 3. Performance & Latency Evaluation ---")
    evaluate_model(model, X_test, y_test)
    print("\nEnd-to-End Pipeline Execution Completed Successfully!")
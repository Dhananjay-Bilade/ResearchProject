import time
from sklearn.metrics import average_precision_score, recall_score

def evaluate_model(model, X_test, y_test):
    # Single Transaction Latency
    single_sample = X_test.iloc[[0]]
    start = time.time()
    for _ in range(1000):
        _ = model.predict_proba(single_sample)
    latency_ms = ((time.time() - start) / 1000) * 1000
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\nSingle-Transaction Inference Latency: {latency_ms:.3f} ms")
    print(f"Precision-Recall AUC (PR-AUC)    : {average_precision_score(y_test, y_prob):.4f}")
    print(f"Fraud Recall                     : {recall_score(y_test, y_pred)*100:.2f}%")
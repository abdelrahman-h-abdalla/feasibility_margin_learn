
def binary_predictions(output, threshold=0.0):
    # Convert network output to binary predictions based on a threshold of 0.0
    return (output > threshold).float()

def compute_metrics(preds, targets):
    TP = ((preds == 1) & (targets == 1)).sum().item()
    TN = ((preds == 0) & (targets == 0)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()
    
    total_samples = TP + TN + FP + FN
    accuracy = (TP + TN) / total_samples if total_samples > 0 else 0

    return TP, TN, FP, FN, accuracy
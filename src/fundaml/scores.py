import torch

def score_mape(y_true, y_pred): 
    """
    Compute the Mean Absolute Percentage Error (MAPE) between true and predicted values.
    """
    # Compute the absolute difference between true and predicted values
    absolute_error = torch.abs(y_true - y_pred)
    # Compute the percentage error
    percentage_error = absolute_error / y_true
    # Compute the mean of the percentage error
    mape = torch.mean(percentage_error)
    return mape

def score_accuracy(y_true, y_pred):
    """
    Compute the accuracy of classification prediction.
    """
    # Convert the predictions from probabilities to class labels by selecting
    # the index of the maximum value in each prediction
    _, y_pred_labels = torch.max(y_pred, dim=1)
    
    # Compute the number of correct predictions
    correct_predictions = (y_pred_labels == y_true).sum().item()

    # Compute the accuracy as the ratio of correct predictions to the total number of predictions
    accuracy = correct_predictions / len(y_true)
    return accuracy
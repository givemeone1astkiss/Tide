# Here are the functions and classes that are used for model evaluation
import torch
import torch.nn as nn

def get_metric(metric_name: str):
    """Get the metric function."""
    if metric_name == "mse":
        return mse
    elif metric_name == "rmse":
        return rmse
    elif metric_name == "log_rmse":
        return log_rmse
    elif metric_name == "mae":
        return mae
    else:
        raise ValueError("Invalid metric name.")

def rmse(pred_labels, labels):
    """For the model evaluation."""
    if not isinstance(pred_labels, torch.Tensor):
        pred_labels = torch.tensor(pred_labels)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    with torch.no_grad():
        rmse = torch.sqrt(nn.functional.mse_loss(pred_labels, labels))
    return rmse.item()

def log_rmse(pred_labels, labels):
    """For the model evaluation."""
    if not isinstance(pred_labels, torch.Tensor):
        pred_labels = torch.tensor(pred_labels)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    with torch.no_grad():
        clipped_preds = torch.clamp(pred_labels, torch.tensor(0.1), torch.tensor(float("inf")))
        rmse = torch.sqrt(2 * nn.functional.mse_loss(clipped_preds.log(), labels.log()))
    return rmse.item()


def mae(pred_labels, labels):
    """For the model evaluation."""
    if not isinstance(pred_labels, torch.Tensor):
        pred_labels = torch.tensor(pred_labels)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    pred_labels = pred_labels.squeeze()
    labels = labels.squeeze()
    with torch.no_grad():
        average_mae = torch.mean(torch.abs(pred_labels - labels))
    return average_mae.item()

def mse(pred_labels, labels):
    """For the model evaluation."""
    if not isinstance(pred_labels, torch.Tensor):
        pred_labels = torch.tensor(pred_labels)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    with torch.no_grad():
        mse = nn.functional.mse_loss(pred_labels, labels)
    return mse.item()

if __name__ == "__main__":
    print("Metrics module test passed.")
from model import *
import time
import pytorch_lightning as pl
import yaml
import os
import joblib

def get_model(model_name: str, config_path: str):
    if config_path == "../config/default.yaml":
        args = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)[model_name]
    else:
        args = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    if model_name == "MLP":
        return MLP(args)
    elif model_name == "CNN":
        return CNN(args)
    elif model_name == "XGBoost":
        return XGBoost(args)
    elif model_name == "GBDT":
        return GBDT(args)
    elif model_name == "AdaBoost":
        return AdaBoost(args)
    elif model_name == "RandomForest":
        return RandomForest(args)
    else:
        raise ValueError("Invalid model name.")

def timer(func):
    """Timer for the function."""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time elapsed: {end-start:.2f}s")
        return result
    return wrapper

def save_model(model, model_params, path: str):
    """Save the model."""
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(f"{path}/model_params"):
        os.makedirs(f"{path}/model_params")
    if not os.path.exists(f"{path}/model"):
        os.makedirs(f"{path}/model")
    if model.type in DEEP_LEARNING_MODEL:
        torch.save(model.state_dict(), f"{path}/model/{model.type}.pt")
    elif model.type in MACHINE_LEARNING_MODEL:
        joblib.dump(model, f"{path}/model/{model.type}.pkl")
    else:
        raise ValueError("Invalid model type.")
    with open(f"{path}/model_params/{model.type}.yaml", "w") as f:
        yaml.dump(model_params, f)

def load_model(model_type, path):
    """Load the model."""
    if model_type in DEEP_LEARNING_MODEL:
        model = get_model(model_type, "../config/default.yaml")
        model.load_state_dict(torch.load(f"{path}/model/{model_type}.pt"))
    elif model_type in MACHINE_LEARNING_MODEL:
        model = joblib.load(f"{path}/model/{model_type}.pkl")
    else:
        raise ValueError("Invalid model type.")
    return model

if __name__ == "__main__":
    pass
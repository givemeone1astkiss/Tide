# Here are the global variables that are used in the project
import numpy as np
import torch

# AAS defines the types of amino acids.
AAS: list = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
# NUM_BP_TYPE defines the number of amino acids types.
NUM_AA_TYPE: int = len(AAS)
# DATASET_PATH is the path of the dataset
DATASET: str = "../data/peptides.xlsx"
# DEEP_LEARNING_MODEL is the list of deep learning models
DEEP_LEARNING_MODEL: list = ["MLP", "LSTM", "CNN"]
# MACHINE_LEARNING_MODEL is the list of machine learning models
MACHINE_LEARNING_MODEL: list = ["XGBoost", "GBDT", "RandomForest", "AdaBoost"]
# EVAL_FUNC is the evaluation function
EVAL_FUNC: list = ["rmse", "mae", "log_rmse", "mse"]
# SAVE_MODEL_PATH is the path to save and load models
SAVE_MODEL_PATH: str = "./save"

# Set seed.
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


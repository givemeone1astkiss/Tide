from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Backbone(pl.LightningModule):
    """
    The backbone of the deep learning models.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        self.criterion = nn.MSELoss()
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

class MLP(Backbone):
    """
    Multi-layer perceptron model.
    Args:
        mlp_params (MLPParams): Parameters for the MLP model.
    """

    def __init__(self, mlp_params: MLPParams):
        super().__init__()
        output_size = 1
        input_size = NUM_AA_TYPE * mlp_params["seq_len"]
        self.fc1 = nn.Linear(input_size, mlp_params["hidden_size"])
        self.fc2 = nn.ModuleList([nn.Linear(mlp_params["hidden_size"], mlp_params["hidden_size"]) for _ in range(mlp_params["num_layers"])])
        self.fc3 = nn.Linear(mlp_params["hidden_size"], output_size)
        self.type = "MLP"
        print("---MLP model initiated.---")

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = F.dropout(x, 0.1)
        for fc in self.fc2:
            x = torch.relu(fc(x))
            x = F.dropout(x, 0.1)
        x = self.fc2(x)
        return x



class CNN(Backbone):
    """
    Convolutional neural network model.
    Args:
        cnn_params (CNNParams): Parameters for the CNN model.
    """
    def __init__(self, cnn_params: CNNParams):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(NUM_AA_TYPE, 32, kernel_size) for kernel_size in cnn_params["kernel_sizes"]]
        )
        self.fc1 = nn.Linear(32 * len(cnn_params["kernel_sizes"]), 128)
        self.fc2 = nn.Linear(cnn_params["hidden_dim"], 1)
        self.type = "CNN"
        print("---CNN model initiated.---")

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
from typing import TypedDict, Optional

class MLPParams(TypedDict):
    seq_len: int
    hidden_size: int
    num_layers: Optional[int]

class CNNParams(TypedDict):
    kernel_sizes: list[int]
    hidden_dim: Optional[int]

class LSTMParams(TypedDict):
    hidden_size: int
    num_layers: Optional[int]
    input_size: Optional[int]
    output_size: Optional[int]

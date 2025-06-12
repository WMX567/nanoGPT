"""
    Utilities for running and plotting coordinate checks.
"""

import math
import pandas as pd
import torch

from dataclasses import dataclass

@dataclass
class DataRow:
    layer: int
    width: int 
    layer_type: str # ['input', 'weight', 'bias']
    fDf: str # ['f', 'Df']
    data_type: str # ['norm', 'mean', 'var']
    value: float
    iteration: int
    seed: int

    def as_dict(self):
        return {
            "layer": self.layer,
            "width": self.width,
            "layer_type": self.layer_type,
            "fDf": self.fDf,
            "data_type": self.data_type,
            "value": self.value,
            "iteration": self.iteration,
            "seed": self.seed
        }

def get_hooks(data: dict, key: str) -> tuple:
    """
        Returns persistent dictionary `hook_data` and a forward hook function
    """
    hook_data = {
        "input_natural_norm": [],
        "input_mean": [],
        "input_var": [],
        "input_diff_natural_norm": [],
        "input_diff_mean": [],
        "input_diff_var": [],
        "weight_natural_norm": [],
        "weight_mean": [],
        "weight_var": [],
        "weight_diff_natural_norm": [],
        "weight_diff_mean": [],
        "weight_diff_var": [],
        "bias_natural_norm": [],
        "bias_mean": [],
        "bias_var": [],
        "bias_diff_natural_norm": [],
        "bias_diff_mean": [],
        "bias_diff_var": [],        
    }

    data[key] = hook_data

    # persistent variables should reduce computational load
    last_input = None
    last_weight = None
    last_bias = None

    def forward_hook(module, input, _):
        nonlocal last_input, last_weight, last_bias

        if type(input) is tuple:
            input = input[0]
        
        input_norm = input.norm(p=2, dim=-1).mean() / math.sqrt(input.size(-1))
        input_mean = input.abs().mean()
        input_var = input.var()
        hook_data["input_natural_norm"].append(input_norm.item())
        hook_data["input_mean"].append(input_mean.item())
        hook_data["input_var"].append(input_var.item())

        weight_norm = module.weight.norm() * (math.sqrt(module.weight.size(1) / module.weight.size(0)))
        weight_mean = module.weight.abs().mean()
        weight_var = module.weight.var()
        hook_data["weight_natural_norm"].append(weight_norm.item())
        hook_data["weight_mean"].append(weight_mean.item())
        hook_data["weight_var"].append(weight_var.item())

        if last_input is not None:
            input_diff = input - last_input
            input_diff_norm = input_diff.norm(p=2) / math.sqrt(input.size(-1))
            input_diff_mean = input_diff.abs().mean()
            input_diff_var = input_diff.var()
            hook_data["input_diff_natural_norm"].append(input_diff_norm.item())
            hook_data["input_diff_mean"].append(input_diff_mean.item())
            hook_data["input_diff_var"].append(input_diff_var.item())
    
        last_input = input.clone()

        if last_weight is not None:
            weight_diff = module.weight - last_weight
            weight_diff_norm = weight_diff.norm() * (math.sqrt(module.weight.size(1) / module.weight.size(0)))
            weight_diff_mean = weight_diff.abs().mean()
            weight_diff_var = weight_diff.var()
            hook_data["weight_diff_natural_norm"].append(weight_diff_norm.item())
            hook_data["weight_diff_mean"].append(weight_diff_mean.item())
            hook_data["weight_diff_var"].append(weight_diff_var.item())

        last_weight = module.weight.clone()

    return forward_hook

def new_rows_from_layer_type(layer_data, layer_type: str, width: int, seed: int, layer: str, fDf: str) -> DataRow:
    new_rows = []
    for data_type in ["natural_norm", "mean", "var"]:
        for idx in range(len(layer_data[f"{layer_type}_{data_type}"])):
            layer_type_name = layer_type if not fDf == 'Df' else layer_type.split('_')[0]
            new_rows.append(
                DataRow(
                    layer=layer,
                    width=width,
                    seed=seed,
                    layer_type=layer_type_name,
                    data_type="norm" if data_type == "natural_norm" else data_type,
                    fDf=fDf,
                    value=layer_data[f"{layer_type}_{data_type}"][idx],
                    iteration=idx
                )
            )
    return new_rows

def dataframe_from_data(data: list, width: int, seed: int) -> pd.DataFrame:
    rows = []

    # data is a list of dictionaries, one for each of the layers
    for key, layer_data in data.items():
        rows.extend(new_rows_from_layer_type(layer_data, 'input', width, seed, key, 'f'))
        rows.extend(new_rows_from_layer_type(layer_data, 'weight', width, seed, key, 'f'))
        rows.extend(new_rows_from_layer_type(layer_data, 'input_diff', width, seed, key, 'Df'))
        rows.extend(new_rows_from_layer_type(layer_data, 'weight_diff', width, seed, key, 'Df'))

    return pd.DataFrame(rows)
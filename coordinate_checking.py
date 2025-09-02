"""
    Utilities for running and plotting coordinate checks.
"""

import math
import pandas as pd
import torch

from copy import deepcopy
from dataclasses import dataclass

def spectral_norm_svd(A: torch.Tensor) -> float:
    # return torch.linalg.matrix_norm(A, ord=2)
    return torch.linalg.svdvals(A.to(torch.float32)).max()

def natural_spectral_norm(A: torch.Tensor, return_list=False, transpose_weights=False) -> float | torch.Tensor:
    # return torch.linalg.matrix_norm(A, ord='fro')
    if len(A.shape) == 3:
        norms = []
        for matrix in A:
            if transpose_weights:
                scale_factor = math.sqrt(matrix.size(0) / matrix.size(1))
            else:
                scale_factor = math.sqrt(matrix.size(1) / matrix.size(0)) 
            norms.append(spectral_norm_svd(matrix) * scale_factor)
        if return_list:
            return norms
        else:
            return torch.tensor(norms).mean()
    if transpose_weights:
        return spectral_norm_svd(A) * math.sqrt(A.size(0) / A.size(1))
    else:
        return spectral_norm_svd(A) * math.sqrt(A.size(1) / A.size(0))

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
    depth: int 
    tag: str

    def as_dict(self):
        return {
            "layer": self.layer,
            "width": self.width,
            "layer_type": self.layer_type,
            "fDf": self.fDf,
            "data_type": self.data_type,
            "value": self.value,
            "iteration": self.iteration,
            "seed": self.seed,
            "depth": self.depth,
            "tag": self.tag
        }

def get_moe_hooks(data: dict, key: str, n_experts: int) -> tuple:
    hook_data = {
        "weight_natural_norm": [],
        "weight_mean": [],
        "weight_var": [],
        "weight_diff_natural_norm": [],
        "weight_diff_mean": [],
        "weight_diff_var": [],
    }

    data[f"{key}_fc1"] = deepcopy(hook_data)
    data[f"{key}_fc2"] = deepcopy(hook_data)

    # for i in range(n_experts):
    #     data[f"{key}_fc1.e{i}"] = deepcopy(hook_data)
    #     data[f"{key}_fc2.e{i}"] = deepcopy(hook_data)

    last_fc1_weight = None
    last_fc2_weight = None

    def forward_hook(module, input, _):
        nonlocal last_fc1_weight, last_fc2_weight

        for tag in ['fc1', 'fc2']:
            if hasattr(module, f'{tag}_weight'):
                fc_weight = getattr(module, f'{tag}_weight')

                fc_weight_norm = natural_spectral_norm(fc_weight, transpose_weights=True)
                fc_weight_mean = fc_weight.abs().mean()
                fc_weight_var = fc_weight.var()

                data[f"{key}_{tag}"]["weight_natural_norm"].append(fc_weight_norm.item())
                data[f"{key}_{tag}"]["weight_mean"].append(fc_weight_mean.item())
                data[f"{key}_{tag}"]["weight_var"].append(fc_weight_var.item())

                # for i in range(len(fc_weight_norm)):
                #     ei = fc_weight[i, :, :]
                #     sn = fc_weight_norm[i]
                #     fc_weight_mean = ei.abs().mean()
                #     fc_weight_var = ei.var()
                #     data[f"{key}_{tag}.e{i}"]["weight_natural_norm"].append(sn.item())
                #     data[f"{key}_{tag}.e{i}"]["weight_mean"].append(fc_weight_mean.item())
                #     data[f"{key}_{tag}.e{i}"]["weight_var"].append(fc_weight_var.item())

                if tag == 'fc1':
                    last_fc_weight = last_fc1_weight
                else:
                    last_fc_weight = last_fc2_weight

                if last_fc_weight is not None:
                    fc_weight_diff = fc_weight - last_fc_weight
                    print(f"{key}_{tag} weight diff shape: {fc_weight_diff.shape}")
                    fc_weight_diff_norm = natural_spectral_norm(fc_weight_diff, transpose_weights=True)
                    fc_weight_diff_mean = fc_weight_diff.abs().mean()
                    fc_weight_diff_var = fc_weight_diff.var()

                    data[f"{key}_{tag}"]["weight_diff_natural_norm"].append(fc_weight_diff_norm.item())
                    data[f"{key}_{tag}"]["weight_diff_mean"].append(fc_weight_diff_mean.item())
                    data[f"{key}_{tag}"]["weight_diff_var"].append(fc_weight_diff_var.item())

                    # for i in range(len(fc_weight_diff_norm)):
                    #     ei_diff = fc_weight_diff[i, :, :]
                    #     sn_diff = fc_weight_diff_norm[i]
                    #     fc_weight_diff_mean = ei_diff.abs().mean()
                    #     fc_weight_diff_var = ei_diff.var()
                    #     data[f"{key}_{tag}.e{i}"]["weight_diff_natural_norm"].append(sn_diff.item())
                    #     data[f"{key}_{tag}.e{i}"]["weight_diff_mean"].append(fc_weight_diff_mean.item())
                    #     data[f"{key}_{tag}.e{i}"]["weight_diff_var"].append(fc_weight_diff_var.item())

                if tag == 'fc1':
                    last_fc1_weight = fc_weight.clone()
                else:
                    last_fc2_weight = fc_weight.clone()
        
    return forward_hook

def get_hooks(data: dict, key: str) -> tuple:
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

        if hasattr(module, 'weight'):
            weight = module.weight

            weight_norm = natural_spectral_norm(weight)
            weight_mean = weight.abs().mean()
            weight_var = weight.var()
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

        if last_weight is not None and hasattr(module, 'weight'):
            weight = module.weight
            
            weight_diff = weight - last_weight
            weight_diff_norm = natural_spectral_norm(weight_diff)
            weight_diff_mean = weight_diff.abs().mean()
            weight_diff_var = weight_diff.var()
            hook_data["weight_diff_natural_norm"].append(weight_diff_norm.item())
            hook_data["weight_diff_mean"].append(weight_diff_mean.item())
            hook_data["weight_diff_var"].append(weight_diff_var.item())

        if hasattr(module, 'weight'):
            last_weight = weight.clone()

    return forward_hook

def new_rows_from_layer_type(layer_data, layer_type: str, width: int, seed: int, layer: str, fDf: str, depth: int, tag: str) -> DataRow:
    new_rows = []
    for data_type in ["natural_norm", "mean", "var"]:
        for idx in range(len(layer_data[f"{layer_type}_{data_type}"])):
            layer_type_name = layer_type if not fDf == 'Df' else layer_type.split('_')[0]
            new_rows.append(
                DataRow(
                    layer=layer,
                    width=width,
                    seed=seed,
                    depth=depth,
                    layer_type=layer_type_name,
                    data_type="norm" if data_type == "natural_norm" else data_type,
                    fDf=fDf,
                    value=layer_data[f"{layer_type}_{data_type}"][idx],
                    iteration=idx,
                    tag=tag
                )
            )
    return new_rows

def dataframe_from_data(data: list, width: int, seed: int, depth: int, tag: str) -> pd.DataFrame:
    rows = []

    # data is a list of dictionaries, one for each of the layers
    for key, layer_data in data.items():
        # only do weight and weight_diff for MoE layers
        if 'fc1' not in key and 'fc2' not in key:
            rows.extend(new_rows_from_layer_type(layer_data, 'input', width, seed, key, 'f', depth, tag))
            rows.extend(new_rows_from_layer_type(layer_data, 'input_diff', width, seed, key, 'Df', depth, tag))
        rows.extend(new_rows_from_layer_type(layer_data, 'weight', width, seed, key, 'f', depth, tag))
        rows.extend(new_rows_from_layer_type(layer_data, 'weight_diff', width, seed, key, 'Df', depth, tag))

    return pd.DataFrame(rows)
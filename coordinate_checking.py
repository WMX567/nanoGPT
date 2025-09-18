"""
    Utilities for running and plotting coordinate checks.
"""
import math
import pandas as pd
import torch
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

spectral_norm_dict = {'q': 1.0, 'k': 1.0, 'v': 1.0}

def spectral_norm_svd(A: torch.Tensor) -> float:
    # return torch.linalg.matrix_norm(A, ord=2)
    return torch.linalg.svdvals(A.to(torch.float32)).max()

def natural_spectral_norm(A: torch.Tensor, return_list=False, transpose_weights=False, key=None) -> float | torch.Tensor:

    if key in spectral_norm_dict:
        return spectral_norm_dict[key]

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

def read_data_from_folder(folder_path: str) -> dict:
    """
    Reads pickled data files from the specified folder and returns a dictionary
    with the filename (without extension) as the key and the loaded data as the value.
    """
    data = {}
    folder_path = Path(folder_path)

    for file in folder_path.glob("*.pkl"):
        with open(file, "rb") as f:
            # Load the pickled data
            file_data = pickle.load(f)
            # Use the filename (without extension) as the key
            file_key = file.stem
            data[file_key] = file_data

    return data

def plot_data(df: pd.DataFrame, x: str, y: str, hue: str = None, style: str = None, title: str = "", save_path: str = ""):
    """
    Plots the data using seaborn and matplotlib.
    """
    plt.figure(figsize=(10, 6))
    
    # Create a line plot with markers
    sns.lineplot(data=df, x=x, y=y, hue=hue, style=style, markers=True, dashes=False)

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot as a file, if a save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def read_coord_check_data(folder_path: str) -> pd.DataFrame:
    """Read coordinate check data from a folder containing pickle files."""
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder {folder_path} does not exist")
    
    all_dataframes = []
    
    # Look for pickle files in the folder
    for pickle_file in folder.glob("*.pkl"):
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, pd.DataFrame):
                    all_dataframes.append(data)
                elif isinstance(data, dict):
                    # If it's raw data dictionary, convert to DataFrame
                    # Extract metadata from filename if possible
                    filename = pickle_file.stem
                    parts = filename.split('_')
                    width = 768  # default
                    seed = 0    # default
                    depth = 3   # default
                    tag = 'default'
                    
                    # Try to extract parameters from filename
                    for part in parts:
                        if 'emb' in part:
                            try:
                                width = int(part.replace('emb', ''))
                            except:
                                pass
                        elif 'seed' in part:
                            try:
                                seed = int(part.replace('seed', ''))
                            except:
                                pass
                    
                    df = dataframe_from_data(data, width, seed, depth, tag)
                    all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {pickle_file}: {e}")
            continue
    
    if not all_dataframes:
        print(f"No valid data found in {folder_path}")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

def plot_coord_check_results(df: pd.DataFrame, save_path: str = None):
    """Create coordinate checking plots from the dataframe."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn are required for plotting. Install with: pip install matplotlib seaborn")
        return
    
    if df.empty:
        print("No data to plot")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Coordinate Check Results', fontsize=16)
    
    # Plot 1: Weight norms by layer and width
    ax1 = axes[0, 0]
    weight_data = df[(df['layer_type'] == 'weight') & (df['data_type'] == 'norm') & (df['fDf'] == 'f')]
    if not weight_data.empty:
        for width in weight_data['width'].unique():
            width_data = weight_data[weight_data['width'] == width]
            mean_vals = width_data.groupby('layer')['value'].mean()
            ax1.plot(mean_vals.index, mean_vals.values, marker='o', label=f'width={width}')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Weight Norm')
        ax1.set_title('Weight Norms by Layer and Width')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight differences by iteration
    ax2 = axes[0, 1]
    weight_diff_data = df[(df['layer_type'] == 'weight') & (df['data_type'] == 'norm') & (df['fDf'] == 'Df')]
    if not weight_diff_data.empty:
        for width in weight_diff_data['width'].unique():
            width_data = weight_diff_data[weight_diff_data['width'] == width]
            mean_vals = width_data.groupby('iteration')['value'].mean()
            ax2.plot(mean_vals.index, mean_vals.values, marker='o', label=f'width={width}')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Weight Difference Norm')
        ax2.set_title('Weight Updates by Iteration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    # Plot 3: Input norms by layer and width
    ax3 = axes[1, 0]
    input_data = df[(df['layer_type'] == 'input') & (df['data_type'] == 'norm') & (df['fDf'] == 'f')]
    if not input_data.empty:
        for width in input_data['width'].unique():
            width_data = input_data[input_data['width'] == width]
            mean_vals = width_data.groupby('layer')['value'].mean()
            ax3.plot(mean_vals.index, mean_vals.values, marker='s', label=f'width={width}')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Input Norm')
        ax3.set_title('Input Norms by Layer and Width')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Variance across seeds
    ax4 = axes[1, 1]
    weight_data_f = df[(df['layer_type'] == 'weight') & (df['data_type'] == 'norm') & (df['fDf'] == 'f')]
    if not weight_data_f.empty:
        variance_by_width = []
        widths = sorted(weight_data_f['width'].unique())
        for width in widths:
            width_data = weight_data_f[weight_data_f['width'] == width]
            # Calculate variance across seeds for each layer
            layer_variances = []
            for layer in width_data['layer'].unique():
                layer_data = width_data[width_data['layer'] == layer]
                if len(layer_data) > 1:
                    layer_variances.append(layer_data['value'].var())
            variance_by_width.append(np.mean(layer_variances) if layer_variances else 0)
        
        ax4.bar(range(len(widths)), variance_by_width)
        ax4.set_xlabel('Width')
        ax4.set_ylabel('Variance Across Seeds')
        ax4.set_title('Stability Across Seeds')
        ax4.set_xticks(range(len(widths)))
        ax4.set_xticklabels(widths)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def analyze_folder(folder_path: str = "2025-09-18_12-32-01"):
    """Complete analysis pipeline: read data and create plots."""
    print(f"Reading data from folder: {folder_path}")
    
    # Check if folder exists in current directory or coord-check-impl
    possible_paths = [
        folder_path,
        f"coord-check-impl/{folder_path}",
        f"./coord-check-impl/{folder_path}"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"Could not find folder. Tried: {possible_paths}")
        return
    
    print(f"Found data at: {data_path}")
    
    # Read data
    df = read_coord_check_data(data_path)
    if df.empty:
        print("No data found or could not read data")
        return
    
    print(f"Loaded {len(df)} data points")
    print(f"Unique widths: {sorted(df['width'].unique())}")
    print(f"Unique seeds: {sorted(df['seed'].unique())}")
    print(f"Unique layers: {sorted(df['layer'].unique())}")
    
    # Create plots
    save_path = f"{folder_path}_coord_check_plots.png"
    plot_coord_check_results(df, save_path)
    
    return df

# Example usage:
# df = analyze_folder("2025-09-18_12-32-01")
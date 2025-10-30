import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def plot_mu_transfer(result_dir, ax, title):
    all_files = glob.glob(f"{result_dir}/*.csv")
    if not all_files:
        print(f"No CSV files found in {result_dir}")
        return False
    
    df = pd.concat([pd.read_csv(f) for f in all_files])
    df['log_lr'] = np.log2(df['lr'])
    
    # Filter out -12 log_lr values, only keep from -11 onwards
    df = df[df['log_lr'] >= -11]
    
    # Define colors for μP style (Purple, Blue, Green, Yellow)
    colors = ['#8B4B9D', '#4682B4', '#2E8B57', '#DAA520']
    markers = ['o', 's', 'D', '^']
    widths = sorted(df['width'].unique())
    
    for i, width in enumerate(widths):
        sub = df[df['width'] == width]
        # For each seed and log_lr combination, take the mean of the last 20 values
        # Then average across all seeds for each log_lr
        sub = sub.groupby(['seed', 'log_lr'])['loss'].apply(lambda x: x.tail(10).mean()).reset_index()
        sub = sub.groupby('log_lr')['loss'].mean().reset_index()
        
        # Use μP style colors
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(sub['log_lr'], sub['loss'], 
               label=f'{width}', 
               linewidth=2.5,
               color=color,
               marker=marker,
               markersize=6,
               alpha=0.9)
    
    # Styling to match μP image
    ax.set_xlabel(r'$\log_2$ LearningRate', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend styling
    legend = ax.legend(title='Width', 
                      title_fontsize=11,
                      fontsize=10,
                      frameon=True,
                      fancybox=False,
                      shadow=False,
                      loc='upper left')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    return True

# Create figure with better sizing
fig, axes = plt.subplots(1, 1, figsize=(10, 6))

# Try to plot from CSV files first
success = False
try:
    success = plot_mu_transfer('standard_practice', axes, 'Standard Practice')
    if success:
        print("Successfully plotted from standard_practice directory")
except Exception as e:
    print(f"Error plotting from standard_practice directory: {e}")

if not success:
    try:
        success = plot_mu_transfer('mu_transfer_results', axes, 'Maximal Update Parametrization (μP)')
        if success:
            print("Successfully plotted from mup directory")
    except Exception as e:
        print(f"Error plotting from mup directory: {e}")

if not success:
    try:
        success = plot_mu_transfer('.', axes, 'Training Results')
        if success:
            print("Successfully plotted from current directory")
    except Exception as e:
        print(f"Error plotting from current directory: {e}")

if not success:
    print("No CSV files found in any directory. Please check if CSV files exist.")

plt.tight_layout()
plt.savefig('mu_transfer_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as mu_transfer_plot.png")
plt.show()
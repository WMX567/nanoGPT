import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_mu_transfer(result_dir, ax, title):
    all_files = glob.glob(f"{result_dir}/*.csv")
    for file in all_files:
        df = pd.read_csv(file)
        width = df['width'].iloc[0]
        log2lr = np.log2(df['lr'].iloc[0])
        final_loss = df['loss'].tail(20).mean()
        ax.plot(log2lr, final_loss, 'o', label=f'width={width}' if log2lr == df['log2lr'].min() else "")

    for width in sorted(df['width'].unique()):
        sub = pd.concat([pd.read_csv(f) for f in all_files if f'width{width}_' in f])
        sub = sub.groupby('log2lr')['loss'].apply(lambda x: x.tail(20).mean()).reset_index()
        ax.plot(sub['log2lr'], sub['loss'], label=f'width={width}')
    ax.set_xlabel(r'$\log_2$LearningRate')
    ax.set_ylabel('Training Loss')
    ax.set_title(title)
    ax.legend(title='Width')

fig, axes = plt.subplots(1, 1, figsize=(12, 5))
plot_mu_transfer('mu_transfer_results/mup', axes[0], 'Maximal Update Parametrization')
plt.tight_layout()
plt.show()
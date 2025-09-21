import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def plot_mu_transfer(result_dir, ax, title):
    all_files = glob.glob(f"{result_dir}/*.csv")
    for file in all_files:
        df = pd.read_csv(file)
        width = df['width'].iloc[0]
        log2lr = df['log2lr'].iloc[0]
        # 只取最后若干步的loss均值（收敛loss）
        final_loss = df['loss'].tail(20).mean()
        ax.plot(log2lr, final_loss, 'o', label=f'width={width}' if log2lr == df['log2lr'].min() else "")
    # 连线
    for width in sorted(df['width'].unique()):
        sub = pd.concat([pd.read_csv(f) for f in all_files if f'width{width}_' in f])
        sub = sub.groupby('log2lr')['loss'].apply(lambda x: x.tail(20).mean()).reset_index()
        ax.plot(sub['log2lr'], sub['loss'], label=f'width={width}')
    ax.set_xlabel(r'$\log_2$LearningRate')
    ax.set_ylabel('Training Loss')
    ax.set_title(title)
    ax.legend(title='Width')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_mu_transfer('mu_transfer_results/sp', axes[0], 'Standard Parametrization')
plot_mu_transfer('mu_transfer_results/mup', axes[1], 'Maximal Update Parametrization')
plt.tight_layout()
plt.show()
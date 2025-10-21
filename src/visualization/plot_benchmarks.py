
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set_theme(style="whitegrid")


if __name__ == '__main__':

    JSON_PATH = 'output/benchmark_results_10_20_183504.json'  # Relative to project root

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    rows = []
    for model_file, metrics in data["results"].items():
        row = {"model_file": model_file}
        row.update(metrics)
        rows.append(row)


    df = pd.DataFrame(rows)

    # color palette for distinction
    palette = sns.color_palette("viridis", len(df))

    output_dir = 'output/plots'
    os.makedirs(output_dir, exist_ok=True)


    numeric_cols = df.select_dtypes(include='number').columns


    for i, col in enumerate(numeric_cols):
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df,
            x='model_file',
            y=col,
            hue='model_file',
            palette=palette,
            legend=False
        )
        plt.title(col.replace('_', ' ').capitalize(), fontsize=16)
        plt.ylabel(col.replace('_', ' ').capitalize(), fontsize=14)
        plt.xlabel('Model Variant', fontsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Zoom y-axis to actual value range with a margin
        ymin = df[col].min()
        ymax = df[col].max()
        yrange = ymax - ymin
        plt.ylim(ymin - 0.05 * yrange, ymax + 0.15 * yrange)
        # Add value labels
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, rotation=0, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'benchmark_{col}.png'), dpi=150)
        plt.close()
    print('Plots saved as benchmark_<metric>.png in the current directory.')




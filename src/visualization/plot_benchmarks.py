
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set_theme(style="whitegrid")


if __name__ == '__main__':

    JSON_PATH = 'output/benchmarks/benchmark_results_10_22_154343.json'  # Relative to project root

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    rows = []
    for model_file, metrics in data["results"].items():
        row = {"model_file": model_file}
        row.update(metrics)
        rows.append(row)


    df = pd.DataFrame(rows)

    # Derive grouping from model filename
    def get_group(name: str) -> str:
        name = name.lower()
        if 'float32' in name:
            return 'float32'
        if 'float16' in name:
            return 'float16'
        if 'dynamic' in name:
            return 'dynamic'
        return 'full_quant'

    def get_variant(name: str) -> str:
        name = name.lower()
        if 'latency' in name:
            return 'latency'
        if 'size' in name:
            return 'size'
        return 'base'

    df['group'] = df['model_file'].apply(get_group)
    df['variant'] = df['model_file'].apply(get_variant)

    #  float32 -> float16 -> dynamic -> full_quant
    group_order = ['float32', 'float16', 'dynamic', 'full_quant']
    variant_order = ['base', 'latency', 'size']
    df['_g'] = pd.Categorical(df['group'], categories=group_order, ordered=True)
    df['_v'] = pd.Categorical(df['variant'], categories=variant_order, ordered=True)
    df_sorted = df.sort_values(by=['_g', '_v', 'model_file']).reset_index(drop=True)

    # Color palette per group
    base_palette = sns.color_palette('Set2', 4) 
    palette_map = {
        'float32': base_palette[0],
        'float16': base_palette[1],
        'dynamic': base_palette[2],
        'full_quant': base_palette[3]
    }

    output_dir = 'output/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Only plot real metric columns, exclude helper columns
    helper_cols = {'_g', '_v', 'group', 'variant'}
    numeric_cols = [
        c for c in df_sorted.columns
        if (c not in helper_cols and c != 'model_file' and pd.api.types.is_numeric_dtype(df_sorted[c]))
    ]


    for i, col in enumerate(numeric_cols):
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df_sorted,
            x='model_file',
            y=col,
            hue='group',
            palette=palette_map,
            dodge=False,
            order=df_sorted['model_file'],
        )
        plt.title(col.replace('_', ' ').capitalize(), fontsize=16)
        plt.ylabel(col.replace('_', ' ').capitalize(), fontsize=14)
        plt.xlabel('Model Variant', fontsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Zoom y-axis to actual value range with a slightly smaller margin
        ymin = df_sorted[col].min()
        ymax = df_sorted[col].max()
        yrange = ymax - ymin
        if yrange == 0:
            # fallback margin if all values equal
            margin = 0.1 * (abs(ymax) if ymax != 0 else 1.0)
            plt.ylim(ymin - margin, ymax + margin)
        else:
            plt.ylim(ymin - 0.02 * yrange, ymax + 0.08 * yrange)
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




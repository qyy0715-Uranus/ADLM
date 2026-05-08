import medmnist
from medmnist import INFO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titleweight': 'bold'
})

DATA_FLAG = 'chestmnist'

# Load metadata
info = INFO[DATA_FLAG]
task = info['task']
label_dict = info['label']
class_names = [label_dict[str(i)] for i in range(len(label_dict))]

# Dynamically get the appropriate dataset class
DataClass = getattr(medmnist, info['python_class'])
dataset = DataClass(split='val', download=True, size=224)
total_samples = len(dataset)

img, label = dataset[3]
output_filename = f"{DATA_FLAG}_3rd_image.png"

img.save(output_filename, format="PNG", dpi=(300, 300))

for key, val in label_dict.items():
    print(f"   Class {key}: {val}")

labels_array = dataset.labels

if task == 'multi-label, binary-class':
    counts = labels_array.sum(axis=0)
    none_count = np.sum(labels_array.sum(axis=1) == 0)
else:
    counts = np.bincount(labels_array.flatten(), minlength=len(label_dict))
    none_count = 0

sorted_indices = np.argsort(counts)[::-1]
sorted_counts = list(counts[sorted_indices])
sorted_names = [class_names[i] for i in sorted_indices]

sorted_names.append("None")
sorted_counts.append(none_count)

sorted_percentages = [(count / total_samples) * 100 for count in sorted_counts]

plt.figure(figsize=(14, 7))
bars = plt.bar(sorted_names, sorted_percentages, color='skyblue', edgecolor='black', linewidth=1.5)

bars[-1].set_color('lightgray')
bars[-1].set_edgecolor('black')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(sorted_percentages) * 0.02),
             f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title(f"Class Prevalence in {DATA_FLAG.upper()} (Validation Split, N={total_samples})",
          fontsize=16, fontweight='bold')
plt.ylabel("Percentage of Total Samples (%)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plot_filename = f"{DATA_FLAG}_prevalence_plot.png"
plt.savefig(plot_filename, dpi=300)
plt.show()

if task == 'multi-label, binary-class':
    co_matrix = np.dot(labels_array.T, labels_array)

    single_label_mask = labels_array.sum(axis=1) == 1
    single_labels_only = labels_array[single_label_mask]
    single_label_counts = single_labels_only.sum(axis=0)

    np.fill_diagonal(co_matrix, single_label_counts)

    co_matrix_sorted = co_matrix[sorted_indices][:, sorted_indices]

    matrix_names = [class_names[i] for i in sorted_indices]

    plt.figure(figsize=(14, 11))

    sns.heatmap(co_matrix_sorted, xticklabels=matrix_names, yticklabels=matrix_names,
                cmap="viridis", annot=True, fmt="d", linewidths=.5,
                annot_kws={"weight": "bold", "size": 12})

    plt.title(
        f"Label Co-occurrence Matrix in {DATA_FLAG.upper()}\n(Sorted by Prevalence | Diagonal = Single-Label "
        f"Occurrences)",
        fontsize=16, fontweight='bold', pad=15)

    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tight_layout()

    co_filename = f"{DATA_FLAG}_co_occurrence.png"
    plt.savefig(co_filename, dpi=300)
    plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="deep", font="DejaVu Sans")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

DIRECTORY = "visualisations"
os.makedirs(DIRECTORY, exist_ok=True)

data_cuda = {
    'N': [256, 512, 1024, 2048, 4096],
    'CPU (ms)': [48.209, 437.874, 4083.50, 33397.6, 281866.718],
    'Naive (ms)': [0.581, 0.769, 2.755, 18.862, 164.363],
    'Tiled (ms)': [0.128, 0.354, 1.911, 15.311, 118.860],
    'cuBLAS (ms)': [0.028, 0.043, 0.261, 1.906, 13.990],
    'Speedup (Tiled vs CPU)': [375, 1234, 1849, 2181, 2371],
    'Speedup (Tiled vs Naive)': [4.54, 2.17, 1.44, 1.23, 1.38],
    'Speedup (cuBLAS vs Naive)': [20.75, 17.88, 10.56, 9.90, 11.75]
}
df = pd.DataFrame(data_cuda)

def plot_time_comparison_log() -> None:
    plt.figure(figsize=(12, 7))

    methods = ['CPU (ms)', 'Naive (ms)', 'Tiled (ms)', 'cuBLAS (ms)']
    colors = ['#C73E1D', '#A23B72', '#F18F01', '#2E86AB']

    for method, color in zip(methods, colors):
        sns.lineplot(data=df, x='N', y=method, marker='o', markersize=8,
                     linewidth=2.5, color=color, label=method.replace(' (ms)', ''))

    plt.title('Время выполнения умножения матриц (Логарифмическая шкала)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Размер матрицы (N × N)', fontsize=12)
    plt.ylabel('Время (мс, log scale)', fontsize=12)
    plt.xticks(df['N'])
    plt.yscale('log') # Критически важно для O(N^3) алгоритмов!
    plt.legend(title='Метод вычисления', fontsize=11, title_fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.tight_layout()
    filename = f"{DIRECTORY}/cuda_time_comparison_log.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_gpu_time_zoom() -> None:
    plt.figure(figsize=(12, 6))

    methods_gpu = ['Naive (ms)', 'Tiled (ms)', 'cuBLAS (ms)']
    colors_gpu = ['#A23B72', '#F18F01', '#2E86AB']

    for method, color in zip(methods_gpu, colors_gpu):
        sns.lineplot(data=df, x='N', y=method, marker='s', markersize=8,
                     linewidth=2.5, color=color, label=method.replace(' (ms)', ''))

    plt.title('Сравнение производительности методов на GPU (Линейная шкала)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Размер матрицы (N × N)', fontsize=12)
    plt.ylabel('Время (мс)', fontsize=12)
    plt.xticks(df['N'])
    plt.legend(title='GPU Метод', fontsize=11, title_fontsize=12)

    best_idx = df['cuBLAS (ms)'].idxmax()
    plt.annotate(f'cuBLAS: {df.iloc[best_idx]["cuBLAS (ms)"]:.2f} мс',
                 xy=(df.iloc[best_idx]['N'], df.iloc[best_idx]['cuBLAS (ms)']),
                 xytext=(df.iloc[best_idx]['N'] - 500, df.iloc[best_idx]['cuBLAS (ms)'] * 1.5),
                 fontsize=11, fontweight='bold', color='#2E86AB',
                 arrowprops=dict(arrowstyle="->", color='#2E86AB', lw=1.5))

    plt.tight_layout()
    filename = f"{DIRECTORY}/cuda_gpu_time_zoom.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_speedup() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df['N']))
    width = 0.25

    rects1 = ax.bar(x - width, df['Speedup (Tiled vs CPU)'], width, label='Tiled vs CPU', color='#F18F01')
    rects2 = ax.bar(x, df['Speedup (Tiled vs Naive)'], width, label='Tiled vs Naive', color='#A23B72')
    rects3 = ax.bar(x + width, df['Speedup (cuBLAS vs Naive)'], width, label='cuBLAS vs Naive', color='#2E86AB')

    ax.set_title('Ускорение (Speedup) различных методов', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Размер матрицы (N)', fontsize=12)
    ax.set_ylabel('Ускорение (раз)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['N'])
    ax.legend(fontsize=11)

    ax.bar_label(rects1, padding=3, fontsize=9, fmt='%.0fx')
    ax.bar_label(rects2, padding=3, fontsize=9, fmt='%.2fx')
    ax.bar_label(rects3, padding=3, fontsize=9, fmt='%.1fx')

    plt.tight_layout()
    filename = f"{DIRECTORY}/cuda_speedup.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def main() -> None:
    print("Генерация графиков для Лабораторной работы №3 (CUDA/cuBLAS)...")
    print("=" * 70)
    plot_time_comparison_log()
    plot_gpu_time_zoom()
    plot_speedup()
    print("=" * 70)
    print("✓ Все графики успешно созданы в папке 'visualisations'!")

if __name__ == "__main__":
    main()
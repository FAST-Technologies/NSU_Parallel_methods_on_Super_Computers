import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Настройки шрифтов и стиля для профессионального вида
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="deep", font="DejaVu Sans")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

DIRECTORY = "visualisations"
os.makedirs(DIRECTORY, exist_ok=True)

# ==============================================================================
# ДАННЫЕ ЭКСПЕРИМЕНТА 1 (Блок 16x16 как наиболее сбалансированный)
# ==============================================================================
N_sizes = [256, 512, 1024, 2048, 4096, 8192]

# Данные для Windows (RTX 5060 Ti, Blackwell)
data_win = {
    'N': N_sizes,
    'Naive': [0.170, 0.294, 1.527, 11.323, 92.000, 742.234],
    'Tiled Static': [0.173, 0.285, 1.228, 8.461, 68.598, 549.468],
    'Tiled Dynamic': [0.062, 0.237, 1.571, 12.346, 99.035, 797.006],
    'cuBLAS': [0.053, 0.031, 0.157, 1.065, 8.152, 62.865],
    'Speedup Static': [0.98, 1.03, 1.24, 1.34, 1.34, 1.35],
    'Speedup Dynamic': [2.74, 1.24, 0.97, 0.92, 0.93, 0.93],
    'Speedup cuBLAS': [3.21, 9.48, 9.73, 10.63, 11.29, 11.81],
    'Efficiency Static': [0.38, 0.40, 0.49, 0.52, 0.52, 0.53],
    'Efficiency Dynamic': [1.07, 0.49, 0.38, 0.36, 0.36, 0.36]
}
df_win = pd.DataFrame(data_win)

# Данные для Linux Server (RTX 4000 SFF Ada)
data_linux = {
    'N': N_sizes,
    'Naive': [0.402, 0.598, 2.168, 14.831, 124.803, 1107.796],
    'Tiled Static': [0.037, 0.198, 1.422, 11.139, 101.827, 842.657],
    'Tiled Dynamic': [0.051, 0.287, 2.097, 16.470, 136.733, 1133.926],
    'cuBLAS': [0.019, 0.060, 0.200, 1.266, 9.375, 112.562],
    'Speedup Static': [10.86, 3.02, 1.52, 1.33, 1.23, 1.31],
    'Speedup Dynamic': [7.88, 2.08, 1.03, 0.90, 0.91, 0.98],
    'Speedup cuBLAS': [21.16, 9.97, 10.84, 11.71, 13.31, 9.84],
    'Efficiency Static': [4.24, 1.18, 0.60, 0.52, 0.48, 0.51],
    'Efficiency Dynamic': [3.08, 0.81, 0.40, 0.35, 0.36, 0.38]
}
df_linux = pd.DataFrame(data_linux)

# ==============================================================================
# 1. Время выполнения (Логарифмическая шкала)
# ==============================================================================
def plot_time_comparison(df, platform_name, filename_suffix):
    plt.figure(figsize=(12, 7))
    methods = ['Naive', 'Tiled Static', 'Tiled Dynamic', 'cuBLAS']
    colors = ['#C73E1D', '#F18F01', '#A23B72', '#2E86AB']
    
    for method, color in zip(methods, colors):
        sns.lineplot(data=df, x='N', y=method, marker='o', markersize=8,
                     linewidth=2.5, color=color, label=method)

    plt.title(f'Время выполнения умножения матриц ({platform_name}, блок 16×16)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Размер матрицы (N × N)', fontsize=12)
    plt.ylabel('Время (мс, логарифмическая шкала)', fontsize=12)
    plt.xticks(df['N'])
    plt.yscale('log')
    plt.legend(title='Метод вычисления', fontsize=11, title_fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    
    filename = f"{DIRECTORY}/time_comparison_{filename_suffix}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

# ==============================================================================
# 2. Ускорение (Speedup) относительно Naive
# ==============================================================================
def plot_speedup(df, platform_name, filename_suffix):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df['N']))
    width = 0.25

    rects1 = ax.bar(x - width, df['Speedup Static'], width, label='Tiled Static', color='#F18F01')
    rects2 = ax.bar(x, df['Speedup Dynamic'], width, label='Tiled Dynamic', color='#A23B72')
    rects3 = ax.bar(x + width, df['Speedup cuBLAS'], width, label='cuBLAS', color='#2E86AB')

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Базовая линия (S=1)')
    
    ax.set_title(f'Ускорение относительно Naive ({platform_name}, блок 16×16)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Размер матрицы (N)', fontsize=12)
    ax.set_ylabel('Ускорение (раз)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['N'])
    ax.legend(fontsize=11)

    ax.bar_label(rects1, padding=3, fontsize=9, fmt='%.2fx')
    ax.bar_label(rects2, padding=3, fontsize=9, fmt='%.2fx')
    ax.bar_label(rects3, padding=3, fontsize=9, fmt='%.1fx')
    plt.tight_layout()
    
    filename = f"{DIRECTORY}/speedup_{filename_suffix}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

# ==============================================================================
# 3. Эффективность параллельных вычислений (E = S / P * 100%)
# ==============================================================================
def plot_efficiency(df, platform_name, filename_suffix):
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(data=df, x='N', y='Efficiency Static', marker='s', markersize=8,
                 linewidth=2.5, color='#F18F01', label='Tiled Static')
    sns.lineplot(data=df, x='N', y='Efficiency Dynamic', marker='^', markersize=8,
                 linewidth=2.5, color='#A23B72', label='Tiled Dynamic')

    plt.title(f'Эффективность CUDA-ядер ({platform_name}, блок 16×16)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Размер матрицы (N × N)', fontsize=12)
    plt.ylabel('Эффективность (%)', fontsize=12)
    plt.xticks(df['N'])
    plt.legend(fontsize=11, title='Метод')
    plt.grid(True, ls="--", alpha=0.7)
    plt.tight_layout()
    
    filename = f"{DIRECTORY}/efficiency_{filename_suffix}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

# ==============================================================================
# Главный запуск
# ==============================================================================
def main():
    print("Генерация графиков для Лабораторной работы №3...")
    print("=" * 70)
    
    # Графики для Windows (RTX 5060 Ti)
    plot_time_comparison(df_win, "Windows / RTX 5060 Ti", "win")
    plot_speedup(df_win, "Windows / RTX 5060 Ti", "win")
    plot_efficiency(df_win, "Windows / RTX 5060 Ti", "win")
    
    # Графики для Linux Server (RTX 4000 SFF)
    plot_time_comparison(df_linux, "Linux Server / RTX 4000 SFF", "linux")
    plot_speedup(df_linux, "Linux Server / RTX 4000 SFF", "linux")
    plot_efficiency(df_linux, "Linux Server / RTX 4000 SFF", "linux")
    
    print("=" * 70)
    print("✓ Все 6 графиков успешно созданы в папке 'visualisations'!")
    print("  Рекомендуется вставить в отчёт графики с суффиксом '_win' как основные.")

if __name__ == "__main__":
    main()
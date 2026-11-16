import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

OUT = "visualisations"
os.makedirs(OUT, exist_ok=True)

# Данные (Блок 16x16, Эксперимент 1)
N_sizes = [256, 512, 1024, 2048, 4096, 8192]

# Windows Data
win_naive = [0.170, 0.294, 1.527, 11.323, 92.000, 742.234]
win_static = [0.173, 0.285, 1.228, 8.461, 68.598, 549.468]
win_dynamic = [0.062, 0.237, 1.571, 12.346, 99.035, 797.006]
win_cublas = [0.053, 0.031, 0.157, 1.065, 8.152, 62.865]
win_spd_static = [0.98, 1.03, 1.24, 1.34, 1.34, 1.35]
win_spd_dynamic = [2.74, 1.24, 0.97, 0.92, 0.93, 0.93]
win_spd_cublas = [3.21, 9.48, 9.73, 10.63, 11.29, 11.81]
win_eff_static = [0.38, 0.40, 0.49, 0.52, 0.52, 0.53]
win_eff_dynamic = [1.07, 0.49, 0.38, 0.36, 0.36, 0.36]

def plot_time_log(naive, static, dynamic, cublas, platform, suffix):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(N_sizes, naive, marker='o', color='#C73E1D', linewidth=2.5, markersize=8, label='Naive')
    ax.plot(N_sizes, static, marker='s', color='#F18F01', linewidth=2.5, markersize=8, label='Tiled Static')
    ax.plot(N_sizes, dynamic, marker='^', color='#A23B72', linewidth=2.5, markersize=8, label='Tiled Dynamic')
    ax.plot(N_sizes, cublas, marker='D', color='#2E86AB', linewidth=2.5, markersize=8, label='cuBLAS')

    ax.set_title(f'Время выполнения ({platform}, блок 16×16)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Размер матрицы (N × N)', fontsize=12)
    ax.set_ylabel('Время (мс, log scale)', fontsize=12)
    ax.set_xticks(N_sizes)
    ax.set_yscale('log')
    ax.legend(fontsize=11, title='Метод')
    plt.tight_layout()
    plt.savefig(f"{OUT}/time_comparison_{suffix}.png", bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: time_comparison_{suffix}.png")

def plot_speedup(sp_static, sp_dynamic, sp_cublas, platform, suffix):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(N_sizes))
    width = 0.25

    ax.bar(x - width, sp_static, width, label='Tiled Static', color='#F18F01')
    ax.bar(x, sp_dynamic, width, label='Tiled Dynamic', color='#A23B72')
    ax.bar(x + width, sp_cublas, width, label='cuBLAS', color='#2E86AB')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Базовая линия (S=1)')

    ax.set_title(f'Ускорение относительно Naive ({platform}, блок 16×16)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Размер матрицы (N)', fontsize=12)
    ax.set_ylabel('Ускорение (раз)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(N_sizes)
    ax.legend(fontsize=11)

    for rect in ax.patches:
        ax.annotate(f'{rect.get_height():.2f}x',
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(f"{OUT}/speedup_{suffix}.png", bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: speedup_{suffix}.png")

def plot_efficiency(eff_static, eff_dynamic, platform, suffix):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(N_sizes, eff_static, marker='s', color='#F18F01', linewidth=2.5, markersize=8, label='Tiled Static')
    ax.plot(N_sizes, eff_dynamic, marker='^', color='#A23B72', linewidth=2.5, markersize=8, label='Tiled Dynamic')

    ax.set_title(f'Эффективность CUDA-ядер ({platform}, блок 16×16)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Размер матрицы (N × N)', fontsize=12)
    ax.set_ylabel('Эффективность (%)', fontsize=12)
    ax.set_xticks(N_sizes)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/efficiency_{suffix}.png", bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: efficiency_{suffix}.png")

def main():
    print("Генерация графиков (Matplotlib)...")
    print("=" * 70)
    
    # Windows
    plot_time_log(win_naive, win_static, win_dynamic, win_cublas, "Windows / RTX 5060 Ti", "win")
    plot_speedup(win_spd_static, win_spd_dynamic, win_spd_cublas, "Windows / RTX 5060 Ti", "win")
    plot_efficiency(win_eff_static, win_eff_dynamic, "Windows / RTX 5060 Ti", "win")
    
    print("✓ Графики для Windows созданы!")
    print("=" * 70)

if __name__ == "__main__":
    main()
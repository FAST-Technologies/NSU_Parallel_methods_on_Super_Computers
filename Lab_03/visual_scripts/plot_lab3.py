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

N_sizes = [256, 512, 1024, 2048, 4096]
cpu_time = [48.209, 437.874, 4083.50, 33397.6, 281866.718]
naive_time = [0.581, 0.769, 2.755, 18.862, 164.363]
tiled_time = [0.128, 0.354, 1.911, 15.311, 118.860]
cublas_time = [0.028, 0.043, 0.261, 1.906, 13.990]

speedup_tiled_cpu = [375, 1234, 1849, 2181, 2371]
speedup_tiled_naive = [4.54, 2.17, 1.44, 1.23, 1.38]
speedup_cublas_naive = [20.75, 17.88, 10.56, 9.90, 11.75]

def plot_time_comparison_log() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(N_sizes, cpu_time, marker='o', color='#C73E1D', linewidth=2.5, markersize=8, label='CPU')
    ax.plot(N_sizes, naive_time, marker='s', color='#A23B72', linewidth=2.5, markersize=8, label='Naive (GPU)')
    ax.plot(N_sizes, tiled_time, marker='^', color='#F18F01', linewidth=2.5, markersize=8, label='Tiled (Shared)')
    ax.plot(N_sizes, cublas_time, marker='D', color='#2E86AB', linewidth=2.5, markersize=8, label='cuBLAS')

    ax.set_title('Время выполнения умножения матриц (Логарифмическая шкала)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Размер матрицы (N × N)', fontsize=12)
    ax.set_ylabel('Время (мс, log scale)', fontsize=12)
    ax.set_xticks(N_sizes)
    ax.set_yscale('log')
    ax.legend(fontsize=11, title='Метод вычисления', title_fontsize=12)

    plt.tight_layout()
    filename = f"{OUT}/cuda_time_comparison_log.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_gpu_time_zoom() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(N_sizes, naive_time, marker='s', color='#A23B72', linewidth=2.5, markersize=8, label='Naive')
    ax.plot(N_sizes, tiled_time, marker='^', color='#F18F01', linewidth=2.5, markersize=8, label='Tiled (Shared)')
    ax.plot(N_sizes, cublas_time, marker='D', color='#2E86AB', linewidth=2.5, markersize=8, label='cuBLAS')

    ax.set_title('Сравнение производительности методов на GPU (Линейная шкала)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Размер матрицы (N × N)', fontsize=12)
    ax.set_ylabel('Время (мс)', fontsize=12)
    ax.set_xticks(N_sizes)
    ax.legend(fontsize=11, title='GPU Метод', title_fontsize=12)

    best_n = 4096
    best_time = cublas_time[-1]
    ax.annotate(f'cuBLAS: {best_time:.2f} мс',
                xy=(best_n, best_time),
                xytext=(best_n - 600, best_time * 1.8),
                fontsize=11, fontweight='bold', color='#2E86AB',
                arrowprops=dict(arrowstyle="->", color='#2E86AB', lw=1.5))

    plt.tight_layout()
    filename = f"{OUT}/cuda_gpu_time_zoom.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_speedup() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(N_sizes))
    width = 0.25

    rects1 = ax.bar(x - width, speedup_tiled_cpu, width, label='Tiled vs CPU', color='#F18F01')
    rects2 = ax.bar(x, speedup_tiled_naive, width, label='Tiled vs Naive', color='#A23B72')
    rects3 = ax.bar(x + width, speedup_cublas_naive, width, label='cuBLAS vs Naive', color='#2E86AB')

    ax.set_title('Ускорение (Speedup) различных методов', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Размер матрицы (N)', fontsize=12)
    ax.set_ylabel('Ускорение (раз)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(N_sizes)
    ax.legend(fontsize=11)

    ax.bar_label(rects1, padding=3, fontsize=9, fmt='%.0fx')
    ax.bar_label(rects2, padding=3, fontsize=9, fmt='%.2fx')
    ax.bar_label(rects3, padding=3, fontsize=9, fmt='%.1fx')

    plt.tight_layout()
    filename = f"{OUT}/cuda_speedup.png"
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
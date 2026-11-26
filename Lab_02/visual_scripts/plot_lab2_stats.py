from typing import Dict
import os
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

OUT = "visualisations"
os.makedirs(OUT, exist_ok=True)

# Базовое время для гибридных расчётов (1x1)
BASE_TIME_HYBRID = 449.515

# Актуальные измерения, сетка 700 x 700 x 700, clu.nusc.ru
mpi_procs = [1, 2, 4, 5, 6, 8, 10, 14, 16, 20]
mpi_time = [447.54, 225.088, 120.327, 97.244, 87.0856, 62.925, 52.8933, 38.7981, 33.8328, 28.386]
mpi_speedup = [1.00, 1.99, 3.72, 4.60, 5.14, 7.11, 8.46, 11.54, 13.23, 15.77]
mpi_efficiency = [100.00, 99.41, 92.98, 92.04, 85.65, 88.90, 84.61, 82.39, 82.67, 78.83]

hybrid_configs = ["1x1", "4x1", "2x2", "1x4", "2x4", "4x2", "1x8", "8x1", 
                  "4x4", "2x8", "8x2", "1x16", "16x1", "10x2", "2x10", "1x20", "20x1", "4x5", "5x4"]

def threads(config: str) -> int:
    m, t = config.split("x")
    return int(m) * int(t)

hybrid_threads = [threads(c) for c in hybrid_configs]
hybrid_time = [449.515, 120.83, 114.488, 120.948, 58.9125, 58.5511, 59.8405, 60.7442, 
               32.2551, 33.6694, 33.1977, 34.1534, 33.8328, 27.0973, 27.6525, 35.8655, 31.5936, 32.3676, 33.8976]
hybrid_speedup = [1.00, 3.72, 3.93, 3.72, 7.63, 7.68, 7.51, 7.40, 
                  13.94, 13.35, 13.54, 13.16, 13.29, 16.59, 16.26, 12.53, 14.23, 13.89, 13.26]
hybrid_efficiency = [100.0, 93.01, 98.16, 92.91, 95.38, 95.97, 93.90, 92.50, 
                     87.10, 83.44, 84.63, 82.26, 83.04, 82.94, 81.28, 62.67, 71.14, 69.44, 66.30]

def plot_pure_mpi() -> None:
    """Графики для чистого MPI"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Производительность при использовании только MPI (Pure MPI, clu.nusc.ru)', fontsize=14, fontweight='bold')

    # 1. Время
    axes[0].plot(mpi_procs, mpi_time, marker='o', color='#2E86AB', linewidth=2, markersize=8)
    axes[0].set_title('Время выполнения')
    axes[0].set_xlabel('Число MPI-процессов (P)')
    axes[0].set_ylabel('Время (сек)')
    axes[0].set_xticks(mpi_procs)

    # 2. Ускорение
    axes[1].plot(mpi_procs, mpi_speedup, marker='s', color='#A23B72', linewidth=2, markersize=8, label='Фактическое ускорение')
    axes[1].plot(mpi_procs, mpi_procs, 'k--', linewidth=1.5, alpha=0.6, label='Идеальное линейное (S=P)')
    axes[1].set_title('Ускорение ($S_p$)')
    axes[1].set_xlabel('Число MPI-процессов (P)')
    axes[1].set_ylabel('Ускорение (раз)')
    axes[1].set_xticks(mpi_procs)
    axes[1].set_ylim(0, 18)
    axes[1].legend()

    # 3. Эффективность
    axes[2].plot(mpi_procs, mpi_efficiency, marker='^', color='#F18F01', linewidth=2, markersize=8, label='Фактическая эффективность')
    axes[2].axhline(100, color='k', linestyle='--', linewidth=1.5, alpha=0.6, label='Идеальная (100%)')
    axes[2].set_title('Эффективность ($E_p$)')
    axes[2].set_xlabel('Число MPI-процессов (P)')
    axes[2].set_ylabel('Эффективность (%)')
    axes[2].set_xticks(mpi_procs)
    axes[2].set_ylim(0, 110)
    axes[2].legend()

    plt.tight_layout()
    filename: str = f"{OUT}/pure_mpi_metrics.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_hybrid_time() -> None:
    """График времени для гибридного режима (отсортированный по времени)"""
    # Собираем и сортируем данные
    combined = list(zip(hybrid_configs, hybrid_time))
    combined.sort(key=lambda x: x[1])
    
    configs = [x[0] for x in combined]
    times = [x[1] for x in combined]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(configs, times)
    
    # Находим лучший (первый после сортировки)
    best_config = configs[0]
    best_time = times[0]
    best_idx = 0
    
    bars[best_idx].set_color("#C73E1D")
    ax.annotate(f"Оптимум: {best_config} ({best_time:.4f} с)",
                xy=(best_idx, best_time), xytext=(best_idx, best_time + 2.0),
                ha="center", arrowprops={"arrowstyle": "->", "color": "#C73E1D"}, fontsize=11, fontweight='bold')
                
    ax.set_title('Время выполнения для различных гибридных конфигураций (отсортировано, clu.nusc.ru)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Конфигурация (MPI-процессы × OpenMP-потоки)', fontsize=12)
    ax.set_ylabel('Время выполнения (сек)', fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    filename: str = f"{OUT}/hybrid_time_sorted.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_hybrid_speedup_efficiency() -> None:
    """Графики ускорения и эффективности гибридного режима с аннотациями"""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Ускорение и эффективность гибридной модели (MPI + OpenMP, clu.nusc.ru)', fontsize=14, fontweight='bold')
    
    for i, config in enumerate(hybrid_configs):
        ax[0].scatter(hybrid_threads[i], hybrid_speedup[i], color='#2E86AB', s=60, zorder=3)
        ax[0].annotate(config, (hybrid_threads[i], hybrid_speedup[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
        ax[1].scatter(hybrid_threads[i], hybrid_efficiency[i], color='#A23B72', s=60, zorder=3)
        ax[1].annotate(config, (hybrid_threads[i], hybrid_efficiency[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
                       
    ideal_x = sorted(set(hybrid_threads))
    ax[0].plot(ideal_x, ideal_x, 'k--', linewidth=1.5, alpha=0.5, label='Идеальное (S=P)')
    ax[0].set(title="Ускорение ($S_P$)", xlabel="Общее число потоков (P × T)", ylabel="Ускорение", ylim=(0, 18))
    
    ax[1].axhline(100, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Идеальная (100%)')
    ax[1].set(title="Эффективность ($E_P$)", xlabel="Общее число вычислительных потоков (P × T)",
              ylabel="Эффективность (%)", ylim=(0, 110))
              
    for a in ax:
        a.legend()
        a.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    filename: str = f"{OUT}/hybrid_speedup_efficiency.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

if __name__ == "__main__":
    print(f"Базовое время (1x1): {BASE_TIME_HYBRID:.3f} с.")
    plot_pure_mpi()
    plot_hybrid_time()
    plot_hybrid_speedup_efficiency()
    print("✓ Все графики успешно созданы в папке 'visualisations'!")
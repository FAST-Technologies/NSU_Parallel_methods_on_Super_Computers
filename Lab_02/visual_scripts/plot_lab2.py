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
BASE_TIME = 317.679

# Актуальные измерения, сетка 700 x 700 x 700.
mpi_times: Dict[int, float] = {
    1: 317.679, 2: 179.244, 4: 117.313, 5: 101.279,
    6: 90.2635, 8: 112.668, 10: 113.691,
}
hybrid_times: Dict[str, float] = {
    "1x1": 317.679, "4x1": 117.313, "2x2": 105.580,
    "1x4": 182.647, "2x4": 115.697, "4x2": 99.9782,
    "1x8": 199.593, "8x1": 112.668, "4x4": 112.872,
    "2x8": 105.558, "8x2": 86.8708, "1x16": 191.861,
    "9x2": 80.8495, "2x9": 119.733, "2x10": 156.838, "10x2": 90.986,
    "10x3": 117.312, "10x4": 108.226, "10x5": 114.849,
    "10x10": 135.400, "10x20": 180.174,
}

def threads(config: str) -> int:
    m, t = config.split("x")
    return int(m) * int(t)

mpi_procs = list(mpi_times)
mpi_time = list(mpi_times.values())
mpi_speedup = [BASE_TIME / t for t in mpi_time]
mpi_efficiency = [s / p * 100 for s, p in zip(mpi_speedup, mpi_procs)]
hybrid_configs = list(hybrid_times)
hybrid_threads = [threads(c) for c in hybrid_configs]
hybrid_time = list(hybrid_times.values())
hybrid_speedup = [BASE_TIME / t for t in hybrid_time]
hybrid_efficiency = [s / p * 100 for s, p in zip(hybrid_speedup, hybrid_threads)]

def plot_pure_mpi() -> None:
    """Графики для чистого MPI (Таблица 2)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Производительность при использовании только MPI (Pure MPI)', fontsize=14, fontweight='bold')

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
    items = sorted(hybrid_times.items(), key=lambda item: item[1])
    configs = [x[0] for x in items]
    times = [x[1] for x in items]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(configs, times)
    best = min(range(len(times)), key=times.__getitem__)
    bars[best].set_color("#C73E1D")
    ax.annotate(f"Лучшее: {times[best]:.4f} с ({configs[best]})",
                xy=(best, times[best]), xytext=(best, times[best] * 1.12),
                ha="center", arrowprops={"arrowstyle": "->", "color": "#C73E1D"})
    ax.set_title('Время выполнения для различных гибридных конфигураций (MPI × OpenMP)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Конфигурация (MPI-процессы × OpenMP-потоки)', fontsize=12)
    ax.set_ylabel('Время выполнения (сек)', fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    filename: str = f"{OUT}/hybrid_time_sorted.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_hybrid_speedup_efficiency() -> None:
    """Графики ускорения и эффективности гибридного режима с аннотациями"""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Ускорение и эффективность гибридной модели (MPI + OpenMP)', fontsize=14, fontweight='bold')
    for i, config in enumerate(hybrid_configs):
        ax[0].scatter(hybrid_threads[i], hybrid_speedup[i], color='#2E86AB', s=60, zorder=3)
        ax[0].annotate(config, (hybrid_threads[i], hybrid_speedup[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
        ax[1].scatter(hybrid_threads[i], hybrid_efficiency[i], color='#A23B72', s=60, zorder=3)
        ax[1].annotate(config, (hybrid_threads[i], hybrid_efficiency[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
    ideal_x = sorted(set(hybrid_threads))
    ax[0].plot(ideal_x, ideal_x, 'k--', linewidth=1.5, alpha=0.5, label='Идеальное (S=P)')
    ax[0].set(title="Ускорение ($S_P$)", xlabel="Общее число потоков (P × T)", ylabel="Ускорение", ylim=(0, 5.5))
    ax[1].axhline(100, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Идеальная (100%)')
    ax[1].set(title="Эффективность ($E_P$)", xlabel="Общее число вычислительных потоков (P × T)",
              ylabel="Эффективность (%)", ylim=(0, 110))
    for ax in ax:
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    filename: str = f"{OUT}/hybrid_speedup_efficiency.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

if __name__ == "__main__":
    print(f"Базовое время: {BASE_TIME:.3f} с.")
    plot_pure_mpi()
    plot_hybrid_time()
    plot_hybrid_speedup_efficiency()

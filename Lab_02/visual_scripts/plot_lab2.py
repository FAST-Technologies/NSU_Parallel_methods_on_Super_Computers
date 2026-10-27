from typing import List, Dict, Tuple, Set
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

DIRECTORY = "visualisations"
os.makedirs(DIRECTORY, exist_ok=True)

# Таблица 2: Pure MPI
mpi_procs: List[int] = [1, 2, 4, 6, 8, 10]
mpi_time: List[float]  = [0.9970, 0.5314, 0.3062, 0.2517, 0.3359, 0.3264]
mpi_speedup: List[float] = [1.00, 1.88, 3.26, 3.96, 2.97, 3.053]
mpi_efficiency: List[float] = [100.00, 93.81, 81.40, 66.02, 34.72, 30.55]

# Таблица 3: Hybrid (MPI x OpenMP)
# Данные отсортированы по возрастанию времени для наглядности на графике времени
hybrid_configs: List[str] = [
    "10×20", "10×10", "10×5", "10×4", "10×3",
    "1×8", "1×4", "2×10", "2×9", "2×8", "4×2", "2×4", "8×1",
    "4×1", "2×2", "8×2", "10×2", "9×2", "1×1"
]
hybrid_threads: List[int] = [
    200, 100, 50, 40, 30,
    8, 4, 20, 18, 16, 8, 8, 8,
    4, 4, 16, 20, 18, 1
]
hybrid_time: List[float] = [
    13.236, 5.9080, 2.54809, 1.8232, 1.0502,
    0.5527, 0.5723, 0.3381, 0.3524, 0.3358, 0.3363, 0.3320, 0.3544,
    0.3060, 0.3225, 0.2354, 0.2364, 0.2199, 0.9875
]
hybrid_speedup: List[float] = [
    0.07, 0.17, 0.39, 0.54, 0.94,
    1.79, 1.73, 2.92, 2.80, 2.94, 2.94, 2.97, 2.79,
    3.23, 3.06, 4.19, 4.18, 4.49, 1.00
]
hybrid_efficiency: List[float] = [
    0.04, 0.17, 0.78, 1.35, 3.13,
    22.33, 43.14, 14.60, 15.57, 18.38, 36.70, 37.18, 34.83,
    80.68, 76.55, 26.22, 20.89, 24.95, 100.0
]

# ==============================================================================
# ФУНКЦИИ ПОСТРОЕНИЯ ГРАФИКОВ
# ==============================================================================
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
    filename: str = f"{DIRECTORY}/pure_mpi_metrics.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")


def plot_hybrid_time() -> None:
    """График времени для гибридного режима (отсортированный по времени)"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Строим столбчатую диаграмму или линию с маркерами. Линия лучше показывает тренд.
    ax.plot(hybrid_configs, hybrid_time, marker='o', color='#C73E1D', linewidth=2, markersize=7)

    ax.set_title('Время выполнения для различных гибридных конфигураций (MPI × OpenMP)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Конфигурация (MPI-процессы × OpenMP-потоки)', fontsize=12)
    ax.set_ylabel('Время выполнения (сек)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    best_idx: int = hybrid_time.index(min(hybrid_time))
    ax.plot(hybrid_configs[best_idx], hybrid_time[best_idx], 'go', markersize=12, markeredgecolor='black', label=f'Оптимум: {hybrid_configs[best_idx]} ({hybrid_time[best_idx]} сек)')
    ax.legend()

    plt.tight_layout()
    filename: str = f"{DIRECTORY}/hybrid_time_sorted.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")


def plot_hybrid_speedup_efficiency() -> None:
    """Графики ускорения и эффективности гибридного режима с аннотациями"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Ускорение и эффективность гибридной модели (MPI + OpenMP)', fontsize=14, fontweight='bold')

    # 1. Ускорение
    axes[0].scatter(hybrid_threads, hybrid_speedup, color='#2E86AB', s=60, zorder=3)
    axes[0].plot(hybrid_threads, hybrid_threads, 'k--', linewidth=1.5, alpha=0.5, label='Идеальное (S=P)')
    axes[0].set_title('Ускорение ($S_p$)')
    axes[0].set_xlabel('Общее число вычислительных потоков (P × T)')
    axes[0].set_ylabel('Ускорение (раз)')
    axes[0].set_ylim(0, 5.5)

    # Аннотации для ускорения
    for i, txt in enumerate(hybrid_configs):
        axes[0].annotate(txt, (hybrid_threads[i], hybrid_speedup[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)

    # 2. Эффективность
    axes[1].scatter(hybrid_threads, hybrid_efficiency, color='#A23B72', s=60, zorder=3)
    axes[1].axhline(100, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Идеальная (100%)')
    axes[1].set_title('Эффективность ($E_p$)')
    axes[1].set_xlabel('Общее число вычислительных потоков (P × T)')
    axes[1].set_ylabel('Эффективность (%)')
    axes[1].set_ylim(0, 110)

    # Аннотации для эффективности
    for i, txt in enumerate(hybrid_configs):
        axes[1].annotate(txt, (hybrid_threads[i], hybrid_efficiency[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)

    for ax in axes:
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    filename: str = f"{DIRECTORY}/hybrid_speedup_efficiency.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")


def main():
    print("Генерация графиков для Лабораторной работы №2...")
    print("=" * 60)
    plot_pure_mpi()
    plot_hybrid_time()
    plot_hybrid_speedup_efficiency()
    print("=" * 60)
    print("✓ Все графики успешно созданы в папке 'visualisations'!")

if __name__ == "__main__":
    main()
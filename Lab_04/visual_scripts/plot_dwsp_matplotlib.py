from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt

# НАСТРОЙКИ ОФОРМЛЕНИЯ
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

OUT: str = "visualisations"
os.makedirs(OUT, exist_ok=True)

# ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ
data_40: Dict[int, Tuple[float, int, int]] = {
    1: (2.61635, 0, 0),
    2: (1.36140, 1, 1),
    3: (1.02812, 0, 0),
    4: (0.782607, 2, 2),
    5: (0.699509, 1, 1),
    6: (0.657706, 3, 3),
    8: (0.685611, 0, 0),
    10: (0.482675, 10, 10),
}

data_100_no_over: Dict[int, Tuple[float, int, int]] = {
    1: (3.13988, 0, 0),
    2: (1.60399, 0, 0),
    3: (1.13820, 1, 1),
    4: (0.859253, 18, 18),
    5: (0.715108, 8, 8),
    6: (0.600974, 5, 5),
    8: (0.525355, 19, 19),
    10: (0.432268, 1, 1),
}

data_100_over: Dict[int, Tuple[float, int, int]] = {
    1: (3.13885, 0, 0),
    2: (1.60526, 0, 0),
    3: (1.13856, 1, 1),
    4: (0.904753, 13, 13),
    5: (0.719233, 12, 12),
    6: (0.601315, 5, 5),
    8: (0.506590, 19, 19),
    10: (0.432402, 1, 1),
}

def calculate_metrics(
    data: Dict[int, Tuple[float, int, int]]
) -> Tuple[List[int], List[float], List[float], List[float]]:
    processes: Dict[int, Tuple[float, int, int]] = sorted(data)
    times: List[float] = [data[p][0] for p in processes]
    base_time: float  = data[1][0]
    speedup: List[float] = [
        base_time / t
        for t in times
    ]
    efficiency: List[float]  = [
        s / p * 100
        for s, p in zip(speedup, processes)
    ]
    return processes, times, speedup, efficiency

# ГРАФИК 1: ВРЕМЯ, УСКОРЕНИЕ И ЭФФЕКТИВНОСТЬ
def plot_metrics(
    data: Dict[int, Tuple[float, int, int]],
    title: str,
    filename: str
) -> None:
    processes, times, speedup, efficiency = calculate_metrics(data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold"
    )
    axes[0].plot(
        processes,
        times,
        marker="o",
        color="#2E86AB",
        linewidth=2,
        markersize=8
    )
    axes[0].set_title("Время выполнения")
    axes[0].set_xlabel("Число MPI-процессов (P)")
    axes[0].set_ylabel("Время (сек)")
    axes[0].set_xticks(processes)

    axes[1].plot(
        processes,
        speedup,
        marker="s",
        color="#A23B72",
        linewidth=2,
        markersize=8,
        label="Фактическое ускорение"
    )
    axes[1].plot(
        processes,
        processes,
        "k--",
        linewidth=1.5,
        alpha=0.6,
        label="Идеальное линейное (S=P)"
    )
    axes[1].set_title("Ускорение ($S_P$)")
    axes[1].set_xlabel("Число MPI-процессов (P)")
    axes[1].set_ylabel("Ускорение (раз)")
    axes[1].set_xticks(processes)
    axes[1].legend()

    axes[2].plot(
        processes,
        efficiency,
        marker="^",
        color="#F18F01",
        linewidth=2,
        markersize=8,
        label="Фактическая эффективность"
    )
    axes[2].axhline(
        100,
        color="k",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="Идеальная (100%)"
    )
    axes[2].set_title("Эффективность ($E_P$)")
    axes[2].set_xlabel("Число MPI-процессов (P)")
    axes[2].set_ylabel("Эффективность (%)")
    axes[2].set_xticks(processes)
    axes[2].set_ylim(0, 110)
    axes[2].legend()
    plt.tight_layout()
    path: str = f"{OUT}/{filename}"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранен: {path}")

# ГРАФИК 2: СРАВНЕНИЕ --oversubscribe
def plot_oversubscribe_comparison() -> None:
    processes = sorted(data_100_no_over)
    times_without: List[float] = [
        data_100_no_over[p][0]
        for p in processes
    ]
    times_with: List[float] = [
        data_100_over[p][0]
        for p in processes
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        processes,
        times_without,
        marker="o",
        color="#2E86AB",
        linewidth=2,
        markersize=8,
        label="Без --oversubscribe"
    )
    ax.plot(
        processes,
        times_with,
        marker="s",
        color="#A23B72",
        linewidth=2,
        markersize=8,
        label="С --oversubscribe"
    )
    ax.set_title(
        "Сравнение времени выполнения (100 задач, baseMs=50)",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Число MPI-процессов (P)")
    ax.set_ylabel("Время выполнения (сек)")
    ax.set_xticks(processes)
    ax.legend()
    plt.tight_layout()
    path: str = f"{OUT}/dwsp_oversubscribe_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранен: {path}")

def main() -> None:
    print("Генерация графиков DWSP...")
    print("=" * 70)
    plot_metrics(
        data_40,
        "DWSP: 40 задач, baseMs=100",
        "dwsp_40_jobs_metrics.png"
    )
    plot_metrics(
        data_100_no_over,
        "DWSP: 100 задач, baseMs=50, без --oversubscribe",
        "dwsp_100_jobs_no_oversubscribe_metrics.png"
    )
    plot_metrics(
        data_100_over,
        "DWSP: 100 задач, baseMs=50, с --oversubscribe",
        "dwsp_100_jobs_oversubscribe_metrics.png"
    )
    plot_oversubscribe_comparison()
    print("=" * 70)
    print(f"Все графики сохранены в папке '{OUT}/'.")

if __name__ == "__main__":
    main()

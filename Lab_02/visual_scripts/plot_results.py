from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep", font="DejaVu Sans")
plt.rcParams['axes.unicode_minus'] = False

directory: str = "visualisations"
threads: List[int] = [1, 2, 4, 8, 12, 16, 20]
strategies: List[str] = ["Dynamic, 16", "Static", "Static, 1", "Guided, 16"]

# Данные для collapse(1)
time_c1: Dict[str, List[float]] = {
    "Dynamic, 16": [4.529, 2.386, 1.317, 0.825, 0.690, 0.632, 0.638],
    "Static":      [4.257, 2.404, 2.087, 1.578, 1.520, 1.334, 1.099],
    "Static, 1":   [4.428, 2.372, 1.335, 0.827, 0.714, 0.600, 0.547],
    "Guided, 16":  [4.580, 2.469, 2.328, 1.290, 1.038, 0.826, 0.748]
}
speedup_c1: Dict[str, List[float]] = {
    "Dynamic, 16": [1.00, 1.8982, 3.4389, 5.5232, 6.5638, 7.1661, 7.1775],
    "Static":      [1.00, 1.7708, 2.0398, 2.6977, 2.8007, 3.1912, 3.8735],
    "Static, 1":   [1.00, 1.8668, 3.3169, 5.3543, 6.2017, 7.3800, 8.0951],
    "Guided, 16":  [1.00, 1.8550, 1.9674, 3.5504, 4.4123, 5.5448, 6.1230]
}
efficiency_c1: Dict[str, List[float]] = {
    "Dynamic, 16": [100, 94.91, 85.97, 69.04, 54.70, 44.79, 35.89],
    "Static":      [100, 88.54, 51.00, 33.72, 23.34, 19.95, 19.37],
    "Static, 1":   [100, 93.34, 82.92, 66.80, 51.68, 46.13, 40.48],
    "Guided, 16":  [100, 92.75, 49.19, 44.38, 36.77, 34.66, 30.62]
}

# Данные для collapse(2)
time_c2: Dict[str, List[float]] = {
    "Dynamic, 16": [4.493, 2.441, 1.259, 0.811, 0.695, 0.582, 0.522],
    "Static":      [4.471, 2.489, 1.875, 1.615, 1.521, 1.254, 1.088],
    "Static, 1":   [4.433, 2.438, 1.275, 0.844, 0.698, 0.623, 0.542],
    "Guided, 16":  [4.617, 2.457, 2.288, 1.239, 1.032, 0.820, 0.738]
}
speedup_c2: Dict[str, List[float]] = {
    "Dynamic, 16": [1.00, 1.8406, 3.5687, 5.5401, 6.4648, 7.7199, 8.6073],
    "Static":      [1.00, 1.7963, 2.3845, 2.7684, 2.9395, 3.5654, 4.1094],
    "Static, 1":   [1.00, 1.8183, 3.4769, 5.2524, 6.3510, 7.1156, 8.1790],
    "Guided, 16":  [1.00, 1.8791, 2.0179, 3.7264, 4.4738, 5.6305, 6.2561]
}
efficiency_c2: Dict[str, List[float]] = {
    "Dynamic, 16": [100, 92.03, 89.22, 69.25, 53.87, 48.25, 43.04],
    "Static":      [100, 89.82, 59.61, 34.61, 24.50, 22.28, 20.55],
    "Static, 1":   [100, 90.92, 86.92, 65.65, 52.93, 44.47, 40.90],
    "Guided, 16":  [100, 93.96, 50.45, 46.58, 37.28, 35.19, 31.28]
}

def plot_metrics(collapse_type: str, time_data: Dict[str, List[float]], speedup_data: Dict[str, List[float]], eff_data: Dict[str, List[float]]) -> None:
    """Функция для построения 3-х графиков для заданного типа collapse"""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f'Производительность трассировки лучей (OpenMP, {collapse_type})', fontsize=16, fontweight='bold', y=1.05)

    ax1: plt.Axes = axes[0]
    for strategy in strategies:
        ax1.plot(threads, time_data[strategy], marker='o', linewidth=2, label=strategy)
    ax1.set_title('Время выполнения (с)')
    ax1.set_xlabel('Количество потоков')
    ax1.set_ylabel('Время (секунды)')
    ax1.set_xticks(threads)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2: plt.Axes = axes[1]
    for strategy in strategies:
        ax2.plot(threads, speedup_data[strategy], marker='s', linewidth=2, label=strategy)
    # Линия идеального линейного ускорения
    ax2.plot(threads, threads, 'k--', linewidth=1.5, label='Идеальное ускорение (S_p = P)')
    ax2.set_title('Ускорение ($S_p$)')
    ax2.set_xlabel('Количество потоков')
    ax2.set_ylabel('Ускорение (раз)')
    ax2.set_xticks(threads)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    ax3: plt.Axes = axes[2]
    for strategy in strategies:
        ax3.plot(threads, eff_data[strategy], marker='^', linewidth=2, label=strategy)
    # Линия идеальной эффективности
    ax3.axhline(100, color='black', linestyle='--', linewidth=1.5, label='Идеальная эффективность (100%)')
    ax3.set_title('Эффективность ($E_p$, %)')
    ax3.set_xlabel('Количество потоков')
    ax3.set_ylabel('Эффективность (%)')
    ax3.set_xticks(threads)
    ax3.set_ylim(0, 110)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    filename: str = f"{directory}/results_collapse_{collapse_type.lower().replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Графики для {collapse_type} сохранены как {filename}")
    plt.show()

print("Генерация графиков...")
plot_metrics("collapse(1)", time_c1, speedup_c1, efficiency_c1)
plot_metrics("collapse(2)", time_c2, speedup_c2, efficiency_c2)
print("Готово!")
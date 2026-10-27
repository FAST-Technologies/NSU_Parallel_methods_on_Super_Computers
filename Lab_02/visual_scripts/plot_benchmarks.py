from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

directory: str = "visualisations"
colors: Dict[str, str] = {
    'Dynamic, 16': '#2E86AB',
    'Static': '#A23B72',
    'Static, 1': '#F18F01',
    'Guided, 16': '#C73E1D'
}
markers: Dict[str, str] = {
    'Dynamic, 16': 'o',
    'Static': 's',
    'Static, 1': '^',
    'Guided, 16': 'D'
}

# Данные для collapse(1)
threads: List[int] = [1, 2, 4, 8, 12, 16, 20]

data_collapse1: Set[Dict[str, Dict[str, List[float]]]] = {
    'Dynamic, 16': {
        'time': [4.529, 2.386, 1.317, 0.825, 0.690, 0.632, 0.638],
        'speedup': [1.00, 1.8982, 3.4389, 5.5232, 6.5638, 7.1661, 7.1775],
        'efficiency': [100, 94.91, 85.97, 69.04, 54.70, 44.79, 35.89]
    },
    'Static': {
        'time': [4.257, 2.404, 2.087, 1.578, 1.520, 1.334, 1.099],
        'speedup': [1.00, 1.7708, 2.0398, 2.6977, 2.8007, 3.1912, 3.8735],
        'efficiency': [100, 88.54, 51.00, 33.72, 23.34, 19.95, 19.37]
    },
    'Static, 1': {
        'time': [4.428, 2.372, 1.335, 0.827, 0.714, 0.600, 0.547],
        'speedup': [1.00, 1.8668, 3.3169, 5.3543, 6.2017, 7.3800, 8.0951],
        'efficiency': [100, 93.34, 82.92, 66.80, 51.68, 46.13, 40.48]
    },
    'Guided, 16': {
        'time': [4.580, 2.469, 2.328, 1.290, 1.038, 0.826, 0.748],
        'speedup': [1.00, 1.8550, 1.9674, 3.5504, 4.4123, 5.5448, 6.1230],
        'efficiency': [100, 92.75, 49.19, 44.38, 36.77, 34.66, 30.62]
    }
}

# Данные для collapse(2)
data_collapse2: Set[Dict[str, Dict[str, List[float]]]] = {
    'Dynamic, 16': {
        'time': [4.493, 2.441, 1.259, 0.811, 0.695, 0.582, 0.522],
        'speedup': [1.00, 1.8406, 3.5687, 5.5401, 6.4648, 7.7199, 8.6073],
        'efficiency': [100, 92.03, 89.22, 69.25, 53.87, 48.25, 43.04]
    },
    'Static': {
        'time': [4.471, 2.489, 1.875, 1.615, 1.521, 1.254, 1.088],
        'speedup': [1.00, 1.7963, 2.3845, 2.7684, 2.9395, 3.5654, 4.1094],
        'efficiency': [100, 89.82, 59.61, 34.61, 24.50, 22.28, 20.55]
    },
    'Static, 1': {
        'time': [4.433, 2.438, 1.275, 0.844, 0.698, 0.623, 0.542],
        'speedup': [1.00, 1.8183, 3.4769, 5.2524, 6.351, 7.1156, 8.1790],
        'efficiency': [100, 90.92, 86.92, 65.65, 52.93, 44.47, 40.90]
    },
    'Guided, 16': {
        'time': [4.617, 2.457, 2.288, 1.239, 1.032, 0.820, 0.738],
        'speedup': [1.00, 1.8791, 2.0179, 3.7264, 4.4738, 5.6305, 6.2561],
        'efficiency': [100, 93.96, 50.45, 46.58, 37.28, 35.19, 31.28]
    }
}

def plot_execution_time(data: Set[Dict[str, Dict[str, List[float]]]], title: str, filename: str) -> None:
    """Построение графика времени выполнения"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy, values in data.items():
        ax.plot(threads, values['time'],
                marker=markers[strategy],
                color=colors[strategy],
                linewidth=2,
                markersize=8,
                label=strategy)

    ax.set_xlabel('Количество потоков', fontsize=12)
    ax.set_ylabel('Время выполнения (с)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_speedup(data: Set[Dict[str, Dict[str, List[float]]]], title: str, filename: str) -> None:
    """Построение графика ускорения с идеальной линией"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Идеальное линейное ускорение
    ax.plot(threads, threads, 'k--', linewidth=1.5, label='Идеальное ускорение', alpha=0.5)

    for strategy, values in data.items():
        ax.plot(threads, values['speedup'],
                marker=markers[strategy],
                color=colors[strategy],
                linewidth=2,
                markersize=8,
                label=strategy)

    ax.set_xlabel('Количество потоков', fontsize=12)
    ax.set_ylabel('Ускорение (S_p)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def plot_efficiency(data, title: str, filename: str) -> None:
    """Построение графика эффективности"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Идеальная эффективность 100%
    ax.axhline(y=100, color='k', linestyle='--', linewidth=1.5, label='Идеальная эффективность (100%)', alpha=0.5)

    for strategy, values in data.items():
        ax.plot(threads, values['efficiency'],
                marker=markers[strategy],
                color=colors[strategy],
                linewidth=2,
                markersize=8,
                label=strategy)

    ax.set_xlabel('Количество потоков', fontsize=12)
    ax.set_ylabel('Эффективность (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.set_ylim(0, 110)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def main() -> None:
    print("Генерация графиков для лабораторной работы...")
    print("=" * 60)

    # Графики для collapse(1)
    print("\nГрафики для collapse(1):")
    plot_execution_time(data_collapse1,
                        'Время выполнения при различных стратегиях планирования\n(collapse=1)',
                        f'{directory}/collapse1_time.png')

    plot_speedup(data_collapse1,
                 'Ускорение при различных стратегиях планирования\n(collapse=1)',
                 f'{directory}/collapse1_speedup.png')

    plot_efficiency(data_collapse1,
                    'Эффективность при различных стратегиях планирования\n(collapse=1)',
                    f'{directory}/collapse1_efficiency.png')

    # Графики для collapse(2)
    print("\nГрафики для collapse(2):")
    plot_execution_time(data_collapse2,
                        'Время выполнения при различных стратегиях планирования\n(collapse=2)',
                        f'{directory}/collapse2_time.png')

    plot_speedup(data_collapse2,
                 'Ускорение при различных стратегиях планирования\n(collapse=2)',
                 f'{directory}/collapse2_speedup.png')

    plot_efficiency(data_collapse2,
                    'Эффективность при различных стратегиях планирования\n(collapse=2)',
                    f'{directory}/collapse2_efficiency.png')

    print("\n" + "=" * 60)
    print("✓ Все графики успешно созданы!")
    print("Файлы сохранены в текущей директории:")
    print(f"  - '{directory}/collapse1_time.png")
    print(f"  - '{directory}/collapse1_speedup.png")
    print(f"  - '{directory}/collapse1_efficiency.png")
    print(f"  - '{directory}/collapse2_time.png")
    print(f"  - '{directory}/collapse2_speedup.png")
    print(f"  - '{directory}/collapse2_efficiency.png")

if __name__ == "__main__":
    main()
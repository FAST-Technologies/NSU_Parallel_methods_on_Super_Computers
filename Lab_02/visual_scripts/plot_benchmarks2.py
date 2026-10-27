from typing import List, Dict, Tuple, Set
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="whitegrid", palette="deep", font="DejaVu Sans")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

DIRECTORY: str = "visualisations"
os.makedirs(DIRECTORY, exist_ok=True)

# Таблица 2: Pure MPI
data_mpi: Dict[str, List[float]] = {
    'Processes': [1, 2, 4, 6, 8, 10],
    'Time': [0.9970, 0.5314, 0.3062, 0.2517, 0.3359, 0.3264],
    'Speedup': [1.00, 1.88, 3.26, 3.96, 2.97, 3.053],
    'Efficiency': [100.00, 93.81, 81.40, 66.02, 34.72, 30.55]
}
df_mpi: pd.DataFrame = pd.DataFrame(data_mpi)

# Таблица 3: Hybrid (MPI x OpenMP)
data_hybrid: Dict[str, List[Any]] = {
    'Config': ["1x1", "4x1", "2x2", "1x4", "2x4", "4x2", "1x8", "8x1",
               "4x4", "2x8", "8x2", "9x2", "2x9", "10x2", "2x10",
               "10x3", "10x4", "10x5", "10x10", "10x20"],
    'Total_Threads': [1, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 18, 18, 20, 20, 30, 40, 50, 100, 200],
    'Time': [0.9875, 0.3060, 0.3225, 0.5723, 0.3320, 0.3363, 0.5527, 0.3544,
             0.2335, 0.3358, 0.2354, 0.2199, 0.3524, 0.2364, 0.3381,
             1.0502, 1.8232, 2.54809, 5.9080, 13.236],
    'Speedup': [1.00, 3.23, 3.06, 1.73, 2.97, 2.94, 1.79, 2.79,
                4.23, 2.94, 4.19, 4.49, 2.80, 4.18, 2.92,
                0.94, 0.54, 0.39, 0.17, 0.07],
    'Efficiency': [100.0, 80.68, 76.55, 43.14, 37.18, 36.70, 22.33, 34.83,
                   26.43, 18.38, 26.22, 24.95, 15.57, 20.89, 14.60,
                   3.13, 1.35, 0.78, 0.17, 0.04]
}
df_hybrid: pd.DataFrame = pd.DataFrame(data_hybrid)
df_hybrid_sorted: pd.DataFrame = df_hybrid.sort_values(by='Time').reset_index(drop=True)

# ==============================================================================
# ГРАФИК 1: Pure MPI (3 метрики в ряд)
# ==============================================================================
def plot_pure_mpi_seaborn() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Производительность при использовании только MPI (Pure MPI)', fontsize=16, fontweight='bold', y=1.05)

    # 1. Время
    sns.lineplot(data=df_mpi, x='Processes', y='Time', marker='o', markersize=8, linewidth=2, color='#2E86AB', ax=axes[0])
    axes[0].set_title('Время выполнения')
    axes[0].set_xlabel('Число MPI-процессов (P)')
    axes[0].set_ylabel('Время (сек)')
    axes[0].set_xticks(df_mpi['Processes'])

    # 2. Ускорение
    sns.lineplot(data=df_mpi, x='Processes', y='Speedup', marker='s', markersize=8, linewidth=2, color='#A23B72', ax=axes[1], label='Фактическое')
    axes[1].plot(df_mpi['Processes'], df_mpi['Processes'], 'k--', linewidth=1.5, alpha=0.6, label='Идеальное (S=P)')
    axes[1].set_title('Ускорение ($S_p$)')
    axes[1].set_xlabel('Число MPI-процессов (P)')
    axes[1].set_ylabel('Ускорение (раз)')
    axes[1].set_xticks(df_mpi['Processes'])
    axes[1].legend()

    # 3. Эффективность
    sns.lineplot(data=df_mpi, x='Processes', y='Efficiency', marker='^', markersize=8, linewidth=2, color='#F18F01', ax=axes[2], label='Фактическая')
    axes[2].axhline(100, color='k', linestyle='--', linewidth=1.5, alpha=0.6, label='Идеальная (100%)')
    axes[2].set_title('Эффективность ($E_p$)')
    axes[2].set_xlabel('Число MPI-процессов (P)')
    axes[2].set_ylabel('Эффективность (%)')
    axes[2].set_xticks(df_mpi['Processes'])
    axes[2].set_ylim(0, 110)
    axes[2].legend()

    plt.tight_layout()
    filename: str = f"{DIRECTORY}/seaborn_pure_mpi.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

# ==============================================================================
# ГРАФИК 2: Гибридный режим - Время выполнения (Barplot, отсортированный)
# ==============================================================================
def plot_hybrid_time_seaborn() -> None:
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=df_hybrid_sorted, x='Config', y='Time', palette='viridis')

    plt.title('Время выполнения для различных гибридных конфигураций (отсортировано)', fontsize=16, fontweight='bold')
    plt.xlabel('Конфигурация (MPI-процессы × OpenMP-потоки)', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    best_row = df_hybrid_sorted.iloc[0]
    best_idx: int = 0
    ax.get_children()[best_idx].set_color('#C73E1D')

    plt.annotate(f'Оптимум: {best_row["Time"]:.4f} сек',
                 xy=(best_idx, best_row['Time']),
                 xytext=(best_idx, best_row['Time'] + 0.5),
                 ha='center', fontsize=11, fontweight='bold', color='#C73E1D',
                 arrowprops=dict(arrowstyle="->", color='#C73E1D'))

    plt.tight_layout()
    filename: str = f"{DIRECTORY}/seaborn_hybrid_time_sorted.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

# ==============================================================================
# ГРАФИК 3: Гибридный режим - Ускорение и Эффективность (Scatter + Line)
# ==============================================================================
def plot_hybrid_metrics_seaborn() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Масштабируемость гибридной модели (MPI + OpenMP)', fontsize=16, fontweight='bold', y=1.05)

    # 1. Ускорение
    sns.scatterplot(data=df_hybrid, x='Total_Threads', y='Speedup', hue='Config', s=80, palette='tab20', ax=axes[0], legend=False)
    sns.lineplot(data=df_hybrid, x='Total_Threads', y='Speedup', color='gray', linestyle='--', marker='o', ax=axes[0])
    axes[0].plot(df_hybrid['Total_Threads'], df_hybrid['Total_Threads'], 'k-.', linewidth=1.5, alpha=0.5, label='Идеальное (S=P)')

    axes[0].set_title('Ускорение ($S_p$)')
    axes[0].set_xlabel('Общее число вычислительных потоков (P × T)')
    axes[0].set_ylabel('Ускорение (раз)')

    axes[0].set_ylim(0, 5.5)

    # Аннотации для ускорения
    for i, row in df_hybrid.iterrows():
        if row['Total_Threads'] <= 20 or row['Config'] == '9x2':
            axes[0].annotate(row['Config'], (row['Total_Threads'], row['Speedup']),
                             textcoords="offset points", xytext=(6, 4), fontsize=8, alpha=0.8)
    axes[0].legend(title='Конфигурация', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Эффективность
    sns.scatterplot(data=df_hybrid, x='Total_Threads', y='Efficiency', hue='Config', s=80, palette='tab20', ax=axes[1], legend=False)
    sns.lineplot(data=df_hybrid, x='Total_Threads', y='Efficiency', color='gray', linestyle='--', marker='^', ax=axes[1])
    axes[1].axhline(100, color='k', linestyle='-.', linewidth=1.5, alpha=0.5, label='Идеальная (100%)')

    axes[1].set_title('Эффективность ($E_p$)')
    axes[1].set_xlabel('Общее число вычислительных потоков (P × T)')
    axes[1].set_ylabel('Эффективность (%)')
    axes[1].set_ylim(0, 110)

    # Аннотации для эффективности
    for i, row in df_hybrid.iterrows():
        if row['Total_Threads'] <= 20 or row['Config'] == '9x2':
            axes[1].annotate(row['Config'], (row['Total_Threads'], row['Efficiency']),
                             textcoords="offset points", xytext=(6, 4), fontsize=8, alpha=0.8)

    plt.tight_layout()
    filename: str = f"{DIRECTORY}/seaborn_hybrid_metrics.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: {filename}")

def main() -> None:
    print("Генерация графиков с использованием Seaborn для Лабораторной работы №2...")
    print("=" * 70)
    plot_pure_mpi_seaborn()
    plot_hybrid_time_seaborn()
    plot_hybrid_metrics_seaborn()
    print("=" * 70)
    print("✓ Все графики успешно созданы в папке 'visualisations'!")
    print("  (Ищите файлы с префиксом 'seaborn_')")

if __name__ == "__main__":
    main()
from typing import List, Dict, Tuple, Set, Any
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

# ==============================================================================
# НОВЫЕ ДАННЫЕ: Сервер nks-1p.sscc.ru (BROADWELL), сетка 700 x 700 x 700
# ==============================================================================

# Таблица 8: Чистый MPI
data_mpi: Dict[str, List[float]] = {
    'Processes': [1, 2, 4, 5, 8, 10, 14, 16, 20],
    'Time': [1272.1, 646.332, 333.575, 288.278, 186.257, 157.547, 116.441, 99.896, 82.4689],
    'Speedup': [1.00, 1.97, 3.81, 4.41, 6.83, 8.07, 10.92, 12.73, 15.43],
    'Efficiency': [100.00, 98.41, 95.34, 88.26, 85.37, 80.74, 78.03, 79.59, 77.13]
}

# Таблица 9: Гибридный режим (MPI x OpenMP)
# Пропущены конфигурации без данных: 16x1, 1x20, 20x1, 4x5, 5x4
data_hybrid: Dict[str, List[Any]] = {
    'Config': ["1x1", "4x1", "2x2", "1x4", "2x4", "4x2", "1x8", "8x1", 
               "4x4", "2x8", "8x2", "1x16", "16x1", "10x2", "2x10", 
               "1x20", "20x1", "4x5", "5x4", "10x3", "10x4"],
    'Total_Threads': [1, 4, 4, 4, 8, 8, 8, 8, 
                      16, 16, 16, 16, 16, 20, 20, 
                      20, 20, 20, 20, 30, 40],
    'Time': [1282.07, 332.596, 331.323, 332.623, 188.843, 185.972, 188.529, 184.086, 
             103.239, 100.207, 104.128, 103.3, 99.3393, 87.7377, 87.7267, 
             88.3322, 82.7234, 81.7758, 84.5373, 63.2458, 61.8568],
    'Speedup': [1.00, 3.85, 3.87, 3.85, 6.79, 6.89, 6.80, 6.96, 
                12.42, 12.79, 12.31, 12.41, 12.91, 14.61, 14.61, 
                14.51, 15.50, 15.68, 15.17, 20.27, 20.73],
    'Efficiency': [100.00, 96.37, 96.74, 96.36, 84.86, 86.17, 85.00, 87.06, 
                   77.62, 79.96, 76.95, 77.57, 80.66, 73.06, 73.07, 
                   72.57, 77.49, 78.39, 75.83, 67.57, 51.82]
}

# ==============================================================================
# НОВЫЕ ДАННЫЕ: Сервер clu.nusc.ru, dl560g10, сетка 700 x 700 x 700
# ==============================================================================

# Таблица 6: Чистый MPI
# data_mpi: Dict[str, List[float]] = {
#     'Processes': [1, 2, 4, 5, 6, 8, 10, 14, 16, 20],
#     'Time': [447.54, 225.088, 120.327, 97.244, 87.0856, 62.925, 52.8933, 38.7981, 33.8328, 28.386],
#     'Speedup': [1.00, 1.99, 3.72, 4.60, 5.14, 7.11, 8.46, 11.54, 13.23, 15.77],
#     'Efficiency': [100.00, 99.41, 92.98, 92.04, 85.65, 88.90, 84.61, 82.39, 82.67, 78.83]
# }


df_mpi: pd.DataFrame = pd.DataFrame(data_mpi)

# Таблица 7: Гибридный режим (MPI x OpenMP)
# data_hybrid: Dict[str, List[Any]] = {
#     'Config': ["1x1", "4x1", "2x2", "1x4", "2x4", "4x2", "1x8", "8x1", 
#                "4x4", "2x8", "8x2", "1x16", "16x1", "10x2", "2x10", "1x20", "20x1", "4x5", "5x4"],
#     'Total_Threads': [1, 4, 4, 4, 8, 8, 8, 8, 
#                       16, 16, 16, 16, 16, 20, 20, 20, 20, 20, 20],
#     'Time': [449.515, 120.83, 114.488, 120.948, 58.9125, 58.5511, 59.8405, 60.7442, 
#              32.2551, 33.6694, 33.1977, 34.1534, 33.8328, 27.0973, 27.6525, 35.8655, 31.5936, 32.3676, 33.8976],
#     'Speedup': [1.00, 3.72, 3.93, 3.72, 7.63, 7.68, 7.51, 7.40, 
#                 13.94, 13.35, 13.54, 13.16, 13.29, 16.59, 16.26, 12.53, 14.23, 13.89, 13.26],
#     'Efficiency': [100.0, 93.01, 98.16, 92.91, 95.38, 95.97, 93.90, 92.50, 
#                    87.10, 83.44, 84.63, 82.26, 83.04, 82.94, 81.28, 62.67, 71.14, 69.44, 66.30]
# }
df_hybrid: pd.DataFrame = pd.DataFrame(data_hybrid)
df_hybrid_sorted: pd.DataFrame = df_hybrid.sort_values(by='Time').reset_index(drop=True)

# ==============================================================================
# ГРАФИК 1: Pure MPI (3 метрики в ряд)
# ==============================================================================
def plot_pure_mpi_seaborn() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Производительность при использовании только MPI (Pure MPI, nks-1p.sscc.ru)', fontsize=16, fontweight='bold', y=1.05)

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
    plt.figure(figsize=(16, 8))  # Увеличили размер для 21 конфигурации
    ax = sns.barplot(data=df_hybrid_sorted, x='Config', y='Time', palette='viridis')

    plt.title('Время выполнения для различных гибридных конфигураций (отсортировано, nks-1p.sscc.ru)', fontsize=16, fontweight='bold')
    plt.xlabel('Конфигурация (MPI-процессы × OpenMP-потоки)', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    best_row = df_hybrid_sorted.iloc[0]
    best_idx: int = 0
    ax.get_children()[best_idx].set_color('#C73E1D')

    plt.annotate(f'Оптимум: {best_row["Config"]} ({best_row["Time"]:.4f} с)',
                 xy=(best_idx, best_row['Time']),
                 xytext=(best_idx, best_row['Time'] + 30),
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
    fig.suptitle('Масштабируемость гибридной модели (MPI + OpenMP, nks-1p.sscc.ru)', fontsize=16, fontweight='bold', y=1.05)

    # 1. Ускорение
    sns.scatterplot(data=df_hybrid, x='Total_Threads', y='Speedup', hue='Config', s=80, palette='tab20', ax=axes[0], legend=False)
    sns.lineplot(data=df_hybrid, x='Total_Threads', y='Speedup', color='gray', linestyle='--', marker='o', ax=axes[0])
    axes[0].plot(df_hybrid['Total_Threads'], df_hybrid['Total_Threads'], 'k-.', linewidth=1.5, alpha=0.5, label='Идеальное (S=P)')

    axes[0].set_title('Ускорение ($S_p$)')
    axes[0].set_xlabel('Общее число вычислительных потоков (P × T)')
    axes[0].set_ylabel('Ускорение (раз)')
    axes[0].set_ylim(0, 22)

    # Аннотации для ускорения (подсвечиваем лучшие конфигурации)
    for i, row in df_hybrid.iterrows():
        if row['Config'] in ['10x3', '10x4', '4x5', '20x1', '16x1']:
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
        if row['Config'] in ['10x3', '10x4', '4x5', '20x1', '16x1', '2x2']:
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
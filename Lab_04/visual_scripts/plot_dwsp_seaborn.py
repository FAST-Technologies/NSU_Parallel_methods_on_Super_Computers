from typing import Dict, List, Tuple
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

sns.set_theme(
    style="whitegrid",
    palette="deep",
    font="DejaVu Sans"
)

DIRECTORY: str = "visualisations"
os.makedirs(DIRECTORY, exist_ok=True)

# ДАННЫЕ DWSP: число MPI-процессов, время, отправлено, получено
data_40: Dict[str, List[float]] = {
    "Processes": [1, 2, 3, 4, 5, 6, 8, 10],
    "Time": [
        2.61635, 1.36140, 1.02812, 0.782607,
        0.699509, 0.657706, 0.685611, 0.482675
    ],
    "Sent": [0, 1, 0, 2, 1, 3, 0, 10],
    "Received": [0, 1, 0, 2, 1, 3, 0, 10]
}

data_100_no_over: Dict[str, List[float]] = {
    "Processes": [1, 2, 3, 4, 5, 6, 8, 10],
    "Time": [
        3.13988, 1.60399, 1.13820, 0.859253,
        0.715108, 0.600974, 0.525355, 0.432268
    ],
    "Sent": [0, 0, 1, 18, 8, 5, 19, 1],
    "Received": [0, 0, 1, 18, 8, 5, 19, 1]
}

data_100_over: Dict[str, List[float]] = {
    "Processes": [1, 2, 3, 4, 5, 6, 8, 10],
    "Time": [
        3.13885, 1.60526, 1.13856, 0.904753,
        0.719233, 0.601315, 0.506590, 0.432402
    ],
    "Sent": [0, 0, 1, 13, 12, 5, 19, 1],
    "Received": [0, 0, 1, 13, 12, 5, 19, 1]
}

def prepare_dataframe(data: Dict[str, List[float]]) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(data)
    base_time: float = df.loc[df["Processes"] == 1, "Time"].iloc[0]
    df["Speedup"] = base_time / df["Time"]
    df["Efficiency"] = (
            df["Speedup"] / df["Processes"] * 100
    )
    return df

df_40: pd.DataFrame = prepare_dataframe(data_40)
df_100_no_over: pd.DataFrame = prepare_dataframe(data_100_no_over)
df_100_over: pd.DataFrame = prepare_dataframe(data_100_over)

# ГРАФИКИ ВРЕМЕНИ, УСКОРЕНИЯ И ЭФФЕКТИВНОСТИ
def plot_metrics(
    df: pd.DataFrame,
    title: str,
    filename: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=1.05
    )
    sns.lineplot(
        data=df,
        x="Processes",
        y="Time",
        marker="o",
        markersize=8,
        linewidth=2,
        color="#2E86AB",
        ax=axes[0]
    )
    axes[0].set_title("Время выполнения")
    axes[0].set_xlabel("Число MPI-процессов (P)")
    axes[0].set_ylabel("Время (сек)")
    axes[0].set_xticks(df["Processes"])

    sns.lineplot(
        data=df,
        x="Processes",
        y="Speedup",
        marker="s",
        markersize=8,
        linewidth=2,
        color="#A23B72",
        ax=axes[1],
        label="Фактическое"
    )
    axes[1].plot(
        df["Processes"],
        df["Processes"],
        "k--",
        linewidth=1.5,
        alpha=0.6,
        label="Идеальное (S=P)"
    )
    axes[1].set_title("Ускорение ($S_P$)")
    axes[1].set_xlabel("Число MPI-процессов (P)")
    axes[1].set_ylabel("Ускорение (раз)")
    axes[1].set_xticks(df["Processes"])
    axes[1].legend()

    sns.lineplot(
        data=df,
        x="Processes",
        y="Efficiency",
        marker="^",
        markersize=8,
        linewidth=2,
        color="#F18F01",
        ax=axes[2],
        label="Фактическая"
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
    axes[2].set_xticks(df["Processes"])
    axes[2].set_ylim(0, 110)
    axes[2].legend()
    plt.tight_layout()

    path: str = f"{DIRECTORY}/{filename}"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранен: {path}")

# СРАВНЕНИЕ ЗАПУСКОВ С --oversubscribe И БЕЗ НЕГО
def plot_oversubscribe_comparison() -> None:
    df: pd.DataFrame = pd.DataFrame({
        "Processes": data_100_no_over["Processes"],
        "Без --oversubscribe": data_100_no_over["Time"],
        "С --oversubscribe": data_100_over["Time"]
    })

    df_long: pd.DataFrame = df.melt(
        id_vars="Processes",
        var_name="Configuration",
        value_name="Time"
    )
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_long,
        x="Processes",
        y="Time",
        hue="Configuration",
        style="Configuration",
        markers=True,
        dashes=False,
        linewidth=2,
        markersize=8
    )
    plt.title(
        "Сравнение времени выполнения (100 задач, baseMs=50)",
        fontsize=16,
        fontweight="bold"
    )
    plt.xlabel("Число MPI-процессов (P)")
    plt.ylabel("Время выполнения (сек)")
    plt.xticks(data_100_no_over["Processes"])
    plt.legend()
    plt.tight_layout()

    path: str = f"{DIRECTORY}/dwsp_oversubscribe_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранен: {path}")

def main() -> None:
    print("Генерация графиков для лабораторной работы DWSP...")
    print("=" * 70)
    plot_metrics(
        df_40,
        "DWSP: 40 задач, baseMs=100",
        "dwsp_40_jobs_metrics.png"
    )
    plot_metrics(
        df_100_no_over,
        "DWSP: 100 задач, baseMs=50, без --oversubscribe",
        "dwsp_100_jobs_no_oversubscribe_metrics.png"
    )
    plot_metrics(
        df_100_over,
        "DWSP: 100 задач, baseMs=50, с --oversubscribe",
        "dwsp_100_jobs_oversubscribe_metrics.png"
    )
    plot_oversubscribe_comparison()
    print("=" * 70)
    print("Все графики сохранены в папке visualisations/")

if __name__ == "__main__":
    main()

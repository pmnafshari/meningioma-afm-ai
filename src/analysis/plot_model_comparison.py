import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time


start_time = time.time()

results_path = "experiments/results.csv"
df = pd.read_csv(results_path)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# accuracy bar chart

plt.figure(figsize=(8,5))

df_sorted = df.sort_values("accuracy", ascending=False)

plt.bar(df_sorted["model"], df_sorted["accuracy"])

plt.ylabel("accuracy")
plt.title("model comparison accuracy")

plt.savefig(output_dir / "accuracy_bar_chart.png")

plt.close()


# model comparison plot

plt.figure(figsize=(8,5))

for model in df["model"].unique():

    subset = df[df["model"] == model]

    plt.plot(
        subset["accuracy"].values,
        label=model
    )

plt.legend()
plt.ylabel("accuracy")
plt.title("model comparison")

plt.savefig(output_dir / "model_comparison_plot.png")

plt.close()


# training time table

end_time = time.time()

time_df = pd.DataFrame({
    "experiment_time_seconds":[end_time-start_time]
})

time_df.to_csv(
    output_dir / "training_time_table.csv",
    index=False
)

print("plots saved in results/")
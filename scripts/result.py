import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

files = {
    "OURS": os.path.join(ROOT, "output", "v1.0(0.2)", "eval_table.csv"),
    "OURS w/ interpolation": os.path.join(ROOT, "output", "v1.1(0.2)", "eval_table.csv"),
    "SOGMP": os.path.join(ROOT, "output", "eval_table.csv"),
}


plt.figure(figsize=(7, 3.5))

for method, csv_path in files.items():
    df = pd.read_csv(csv_path)

    #  sorting n=1..n=10 
    step_cols = sorted([c for c in df.columns if c.startswith("n=")],
                       key=lambda s: int(s.split("=")[1]))

    x = np.arange(1, len(step_cols) + 1)

    mean = df[step_cols].mean(axis=0).to_numpy()

    plt.plot(x, mean, marker="o", linewidth=2, label=method)

plt.xlabel("Prediction time steps")
plt.ylabel("Average IoU")
plt.xticks(x)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
plt.tight_layout()

out_png = os.path.join(ROOT, "output", "iou_eval_plot_0.2.png")
plt.savefig(out_png, dpi=300)
plt.show()
print("Saved:", out_png)
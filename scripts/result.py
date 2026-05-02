import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

files = {
    "scope": os.path.join(ROOT, "output", "compare", "ssim_scope", "eval_table.csv"),
    "scope++": os.path.join(ROOT, "output", "compare", "ssim_scope++", "eval_table.csv"),
    "so_scope": os.path.join(ROOT, "output", "compare", "ssim_so_scope", "eval_table.csv"),
    "ldm2.0.1": os.path.join(ROOT, "output", "compare", "ssim_ldm2.0.1", "eval_table.csv"),
    "ldm2.9.0": os.path.join(ROOT, "output", "compare", "ssim_ldm2.9.0", "eval_table.csv"),
    #"ldm2.0": os.path.join(ROOT, "output", "compare", "ssim_ldm2.0", "eval_table.csv"),
    #"ldm2.0.7": os.path.join(ROOT, "output", "compare", "ssim_ldm2.0.7", "eval_table.csv"),
    #"ldm2.0.8": os.path.join(ROOT, "output", "compare", "ssim_ldm2.0.8", "eval_table.csv"),
    #"ldm2.0.9": os.path.join(ROOT, "output", "compare", "ssim_ldm2.0.9", "eval_table.csv"),
    # "ldm2.0.10": os.path.join(ROOT, "output", "compare", "ssim_ldm2.0.10", "eval_table.csv"),
    # "ldm2.0.11": os.path.join(ROOT, "output", "compare", "ssim_ldm2.0.11", "eval_table.csv"),
    # "ldm2.5": os.path.join(ROOT, "output", "compare", "ssim_ldm2.5", "eval_table.csv"),
    # "ldm2.7": os.path.join(ROOT, "output", "compare", "ssim_ldm2.7", "eval_table.csv"),
    # "ldm2.8": os.path.join(ROOT, "output", "compare", "ssim_ldm2.8", "eval_table.csv"),
    #"v1.6" : os.path.join(ROOT, "output", "compare", "ssim_v1.6", "eval_table.csv"),
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
plt.ylabel("Average SSIM")
plt.xticks(x)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
plt.tight_layout()

out_png = os.path.join(ROOT, "output", "compare", "ssim_compare.png")
plt.savefig(out_png, dpi=500)
plt.show()
print("Saved:", out_png)

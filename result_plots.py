import os
import os.path

import matplotlib.pyplot as plt
import numpy as np

# Set up New Computer Modern font
# matplotlib.rcParams["mathtext.fontset"] = "cm"
# matplotlib.rcParams["font.family"] = "New Computer Modern"

# Set global font size parameters
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }
)

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "output", "charts_svg")
os.makedirs(output_dir, exist_ok=True)

# Data
models = ["Chain-of-Thought", "Dynamic Few-Shot", "Self-Refine"]
avg_scores = [0.814, 0.717, 0.608]
avg_times = [3.045, 6.553, 9.632]
input_tokens = [126657, 454217, 762838]
output_tokens = [164028, 327248, 440144]
num_solved_alone = [3, 8, 5]
num_solved_better = [59, 20, 8]

x = np.arange(len(models))
width = 0.35

# Save each chart as an SVG
# Chart 1: Average Score
plt.figure(figsize=(8, 6))
plt.bar(x, avg_scores, capsize=5, color="skyblue")
plt.xticks(x, models, rotation=20)
plt.ylabel("Average Score")
plt.title("Average Score per Model")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(f"{output_dir}/avg_score.svg", format="svg")

# Chart 2: Average Generation Time
plt.figure(figsize=(8, 6))
plt.bar(x, avg_times, capsize=5, color="salmon")
plt.xticks(x, models, rotation=20)
plt.ylabel("Average Generation Time (s)")
plt.title("Average Generation Time per Model")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(f"{output_dir}/avg_generation_time.svg", format="svg")

# Chart 3: Token Usage
plt.figure(figsize=(8, 6))
plt.bar(
    x,
    input_tokens,
    capsize=5,
    label="Input Tokens",
    color="mediumseagreen",
)
plt.bar(
    x,
    output_tokens,
    bottom=input_tokens,
    capsize=5,
    label="Output Tokens",
    color="mediumorchid",
)
plt.xticks(x, models, rotation=20)
plt.ylabel("Tokens")
plt.title("Token Usage per Model")
plt.legend()
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(f"{output_dir}/token_usage.svg", format="svg")

# Chart 4: Solver Performance
plt.figure(figsize=(8, 6))
plt.bar(
    x,
    num_solved_better,
    capsize=5,
    label="Top Solver",
    color="plum"
)
plt.bar(
    x,
    num_solved_alone,
    bottom=num_solved_better,
    capsize=5,
    label="Only Solver",
    color="darkturquoise",
)
plt.xticks(x, models, rotation=20)
plt.ylabel("Number of Problems")
plt.title("Top vs. Only Solves per Model")
plt.legend()
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(f"{output_dir}/solver_performance.svg", format="svg")


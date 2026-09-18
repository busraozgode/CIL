import subprocess

from cl_framework.config import DATASETS, STRATEGIES, MAX_PARALLEL_JOBS


n_experiments = len(DATASETS) * len(STRATEGIES)

array_range = f"0-{n_experiments - 1}%{MAX_PARALLEL_JOBS}"

print(f"Datasets: {len(DATASETS)}")
print(f"Strategies: {len(STRATEGIES)}")
print(f"Total experiments: {n_experiments}")
print(f"SLURM array: {array_range}")

subprocess.run([
    "sbatch",
    f"--array={array_range}",
    "run_experiment_array.sbatch"
])
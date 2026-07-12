import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from esa_main_framework import run_esa_optimization
from esa_benchmark_functions import FUNC_CONFIG

def plot_modes(all_mode_histories, func_name, dim, save_dir='paper_figures'):
    plt.figure(figsize=(8, 5), dpi=120)
    for mode, histories in all_mode_histories.items():
        min_len = min(len(h) for h in histories)
        running_bests = [np.minimum.accumulate(h[:min_len]) for h in histories]
        avg_line = np.mean(running_bests, axis=0)
        plt.plot(range(1, min_len + 1), avg_line, label=mode, linewidth=2)
    
    plt.yscale('log')
    plt.title(f"Table II Comparison: {func_name.upper()} ({dim}D)")
    plt.xlabel('NFE'); plt.ylabel('Fitness (Log)')
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/table2_{func_name}_{dim}D.png")
    plt.close()

def generate_table_ii():
    functions = ['ellipsoid', 'rosenbrock', 'griewank']
    dimensions = [30, 50, 100]
    modes = ['ES-a1', 'ES-a2', 'ES-a3', 'ES-a4', 'ESA']
    num_runs = 5
    max_nfe = 1000
    
    os.makedirs('paper_tables', exist_ok=True)
    rows = []

    for func_name in functions:
        cfg = FUNC_CONFIG[func_name]
        for dim in dimensions:
            row_data = {'Functions': func_name, 'd': dim}
            all_mode_histories = {}
            for mode in modes:
                histories = []
                results = []
                for run in range(num_runs):
                    best, hist = run_esa_optimization(
                        agent_type="QL", obj_func=cfg['f'], lb_val=cfg['lb'],
                        ub_val=cfg['ub'], dim=dim, max_nfe=max_nfe,
                        seed=2026 + run, mode=mode
                    )
                    results.append(best)
                    histories.append(hist)
                all_mode_histories[mode] = histories
                row_data[f'{mode}_Mean'] = f"{np.mean(results):.2E}"
                row_data[f'{mode}_Std'] = f"{np.std(results):.2E}"
            plot_modes(all_mode_histories, func_name, dim)
            rows.append(row_data)

    pd.DataFrame(rows).to_csv('paper_tables/table_ii_results.csv', index=False)
    print("Table II 完成，圖表已存入 paper_figures/")

if __name__ == "__main__":
    generate_table_ii()
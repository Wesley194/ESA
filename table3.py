import os
import numpy as np
import pandas as pd
from esa_main_framework import run_esa_optimization
from esa_benchmark_functions import FUNC_CONFIG

def generate_table_iii():
    functions = ['ellipsoid', 'rosenbrock', 'griewank']
    dimensions = [30]
    modes = ['ESA-no-a1', 'ESA-no-a2', 'ESA-no-a3', 'ESA-no-a4', 'ESA']
    num_runs = 5
    max_nfe = 1000
    
    os.makedirs('paper_tables', exist_ok=True)
    rows = []

    for func_name in functions:
        cfg = FUNC_CONFIG[func_name]
        for dim in dimensions:
            row_data = {'Functions': func_name, 'd': dim}
            print(f"\n正在跑 Table III -> 函數: {func_name} | 維度: {dim}D")
            
            for mode in modes:
                run_bests = []
                for run in range(num_runs):
                    final_best = run_esa_optimization(
                        agent_type="QL",
                        obj_func=cfg['f'],
                        lb_val=cfg['lb'],
                        ub_val=cfg['ub'],
                        dim=dim,
                        max_nfe=max_nfe,
                        seed=2026 + run,
                        mode=mode
                    )
                    run_bests.append(final_best)
                
                row_data[f'{mode}_Mean'] = f"{np.mean(run_bests):.2E}"
                row_data[f'{mode}_Std'] = f"{np.std(run_bests):.2E}"
                
            rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_csv('paper_tables/table_iii_results.csv', index=False)
    print("\n[成功] Table III 數據已生成至 'paper_tables/table_iii_results.csv'")
    print(df.to_string())

if __name__ == "__main__":
    generate_table_iii()

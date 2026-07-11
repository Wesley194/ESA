import os
import time
import numpy as np
import pandas as pd
from esa_main_framework import run_esa_optimization
from esa_benchmark_functions import FUNC_CONFIG

def run_comprehensive_experiments():
    # 1. 實驗矩陣設定
    functions_to_test = ['ellipsoid', 'rosenbrock', 'griewank']
    dimensions_to_test = [30, 50, 100]  # 測試不同維度
    num_runs = 5                      # 每個組合重複 n 次
    max_nfe = 1000                     # 真實評估上限
    
    os.makedirs('multi_dim_results', exist_ok=True)
    all_combination_summary = []
    
    start_time = time.time()
    
    # 2. 開始巢狀迴圈：函數 -> 維度 -> 重複實驗
    for func_name in functions_to_test:
        for dim in dimensions_to_test:
            print(f"\n[測試中] 函數: {func_name.upper()} | 維度: {dim}D")
            
            run_results = []
            config = FUNC_CONFIG[func_name]
            obj_func = config['f']
            lb_val = config['lb']
            ub_val = config['ub']
            for run in range(num_runs):
                seed = 10 + run  # 固定隨機種子確保實驗可重複
                
                # 執行主程式
                final_best = run_esa_optimization(
                    obj_func=obj_func,
                    lb_val=lb_val,
                    ub_val=ub_val,
                    dim=dim,
                    max_nfe=max_nfe,
                    seed=seed
                )
                run_results.append(final_best)
            
            # 3. 計算該組合的統計指標
            run_results = np.array(run_results)
            mean_val = np.mean(run_results)
            std_val = np.std(run_results)
            best_val = np.min(run_results)
            worst_val = np.max(run_results)
            
            # 記錄至總表
            all_combination_summary.append({
                'Function': func_name,
                'Dimension': f"{dim}D",
                'Best': f"{best_val:.4e}",
                'Worst': f"{worst_val:.4e}",
                'Mean': f"{mean_val:.4e}",
                'Std': f"{std_val:.4e}"
            })
            
            print(f"-> 統計結果: Mean = {mean_val:.4e}, Std = {std_val:.4e}")

    # 4. 匯出最終表格
    df_final = pd.DataFrame(all_combination_summary)
    df_final.to_csv('multi_dim_results/final_paper_table.csv', index=False)
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n==================== 實驗全部結束 ====================")
    print(f"總共耗時: {total_duration/60:.2f} 分鐘")
    print("已生成最終數據報表：'multi_dim_results/final_paper_table.csv'")
    print(df_final.to_string(index=False))

if __name__ == "__main__":
    run_comprehensive_experiments()

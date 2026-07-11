import numpy as np
import pandas as pd
from esa_main_framework import run_esa_optimization
from esa_benchmark_functions import FUNC_CONFIG

if __name__ == "__main__":
    
    target_functions = ['ellipsoid', 'rosenbrock', 'griewank']
    
    # 定義要測試的消融模式
    # 'none' = 基準線(所有特徵都在)
    ablation_modes = [
        'none', 
        's_0', 
        's_1',  
        's_2',   
        's_3',   
        's_4',   
        's_5',   
        'prev_action' # 拔除 One-hot 
    ]
    
    test_seeds = [42, 43, 44, 45, 46]
    
    # 儲存結果
    results = []

    for target_function in target_functions:
        config = FUNC_CONFIG[target_function]
        obj_func = config['f']
        lb_val = config['lb']
        ub_val = config['ub']
        
        print(f"\n{'='*50}")
        print(f"開始測試目標函數: {target_function.upper()}")
        print(f"{'='*50}")
        
        for mode in ablation_modes:
            mode_name = "Baseline (全開)" if mode == 'none' else f"拔除 {mode}"
            print(f"正在執行消融模式: {mode_name}")
            
            run_results = []
            for seed in test_seeds:
                best_val = run_esa_optimization(
                    agent_type="DQN", 
                    obj_func=obj_func, 
                    lb_val=lb_val, 
                    ub_val=ub_val,
                    dim=30, 
                    max_nfe=1000, 
                    seed=seed,
                    ablation_mode=mode 
                )
                run_results.append(best_val)
            
            # 計算平均最佳解與標準差
            avg_best = np.mean(run_results)
            std_best = np.std(run_results)
            min_best = np.min(run_results) 
            max_best = np.max(run_results) 
            
            results.append({
                'Function': target_function,
                'Ablation': mode_name,
                'Mean_Best': avg_best,
                'Std_Best': std_best,
                'Best(Min)': min_best,
                'Worst(Max)': max_best
            })
            print(f"{mode_name} 完成 | 平均最佳解: {avg_best:.4e} ± {std_best:.4e}")

    # 輸出最終結果報表
    print("\n" + "="*50)
    print("消融測試最終結果")
    print("="*50)
    df_results = pd.DataFrame(results)
    
    df_display = df_results.set_index(['Function', 'Ablation'])
    pivot_df = df_results.pivot(index='Ablation', columns='Function', values='Mean_Best')
    print(pivot_df)
    
    df_results.to_csv('dqn_ablation_results.csv', index=False)
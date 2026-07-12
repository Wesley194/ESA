import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from esa_main_framework import run_esa_optimization
from esa_benchmark_functions import FUNC_CONFIG

def main():
    target_functions = ['ellipsoid', 'rosenbrock', 'griewank']
    dimensions = [30, 50, 100]
    agents = ['QL', 'DQN']
    test_seeds = [42,43,44,45,46]  # 使用 3 個 seed 取平均，讓曲線更準確
    max_evals = 1000
    
    # 建立一個資料夾來存放圖表
    os.makedirs('convergence_plots', exist_ok=True)
    
    # 用來存所有測試數據的 list
    all_records = []

    for dim in dimensions:
        print(f"\n{'*'*60}")
        print(f"開始測試維度 Dimension: {dim}")
        print(f"{'*'*60}")
        
        for func_name in target_functions:
            config = FUNC_CONFIG[func_name]
            obj_func = config['f']
            lb_val = config['lb']
            ub_val = config['ub']
            
            print(f"\n目標函數: {func_name.upper()} | 維度: {dim}")
            
            for agent in agents:
                for seed in test_seeds:
                    # 呼叫主程式，取得最終結果與歷史收斂軌跡
                    best_val, history_nfe, history_best = run_esa_optimization(
                        agent_type=agent, 
                        obj_func=obj_func, 
                        lb_val=lb_val, 
                        ub_val=ub_val,
                        dim=dim, 
                        max_nfe=max_evals, 
                        seed=seed,
                        ablation_mode='none' # 基準測試
                    )
                    
                    # 為了讓 QL 和 DQN 可以在每一個 NFE 精準對齊，進行線性內插
                    # 統一將 X 軸切分成 100, 101, 102 ... 1000
                    standard_nfes = np.arange(100, max_evals + 1)
                    interp_bests = np.interp(standard_nfes, history_nfe, history_best)
                    
                    # 將內插後的數據寫入紀錄
                    for n, b in zip(standard_nfes, interp_bests):
                        all_records.append({
                            'Dimension': dim,
                            'Function': func_name,
                            'Agent': agent,
                            'Seed': seed,
                            'NFE': n,
                            'Fitness': b
                        })
                
                print(f"{agent} 測試完成 (共 {len(test_seeds)} 個 seeds)")

    # 將數據儲存到 CSV
    df = pd.DataFrame(all_records)
    csv_filename = 'convergence_history_QL_vs_DQN.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n 所有收斂軌跡已儲存至: {csv_filename}")


    print(" 正在繪製收斂曲線圖...")
    
    # 設定 seaborn 畫圖風格
    sns.set_theme(style="whitegrid")
    
    for dim in dimensions:
        for func_name in target_functions:
            # 篩選特定維度與函數的資料
            subset_df = df[(df['Dimension'] == dim) & (df['Function'] == func_name)]
            
            plt.figure(figsize=(10, 6))
            
            # 使用 seaborn 畫線圖，它會自動將不同 seed 的資料取平均，並畫出陰影 (信賴區間)
            sns.lineplot(
                data=subset_df, 
                x='NFE', 
                y='Fitness', 
                hue='Agent',      # 用 Agent 來區分線的顏色
                style='Agent',    # 用 Agent 來區分線的樣式 (實線/虛線)
                linewidth=2
            )
            
            plt.title(f"Convergence Comparison: {func_name.upper()} (Dim={dim})", fontsize=16, fontweight='bold')
            plt.xlabel("Number of Function Evaluations (NFE)", fontsize=12)
            plt.ylabel("Best Fitness Value", fontsize=12)
            
            # 最佳化演算法的 fitness 通常會逼近 0，使用 Log scale (對數座標) 可以更清楚看到收斂差距
            plt.yscale('log') 
            
            plt.legend(title='Agent Type', fontsize=11, title_fontsize=12)
            plt.tight_layout()
            
            # 儲存圖表
            plot_filename = f"convergence_plots/conv_{func_name}_dim{dim}.png"
            plt.savefig(plot_filename, dpi=300)
            plt.close() # 關閉畫布以釋放記憶體
            
    print(f"收斂比較圖已成功儲存至 'convergence_plots' 資料夾中！")

if __name__ == "__main__":
    main()
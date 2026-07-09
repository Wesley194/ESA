import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from esa_main_framework import run_esa_optimization
from esa_benchmark_functions import FUNC_CONFIG

def run_all_experiments():
    # 1. 實驗參數設定
    functions_to_test = ['ellipsoid', 'rosenbrock', 'griewank']
    dimension = 30
    max_nfe = 1000
    num_runs = 20  # 每種函數獨立重複執行 20 次以取得統計顯著性
    
    # 建立用來存放實驗結果的資料夾
    os.makedirs('experiment_results', exist_ok=True)
    
    # 準備一個總表存放所有函數的統計結果
    summary_data = []

    # 2. 開始對每個 Benchmark 函數跑實驗
    for func_name in functions_to_test:
        print(f"\n==================== 開始測試函數: {func_name} ====================")
        
        all_runs_best_fitness = []  # 記錄每次 Run 的最終最優值
        all_runs_history = []       # 記錄每次 Run 的整條收斂曲線 
    
        config = FUNC_CONFIG[func_name]
        obj_func = config['f']
        lb_val = config['lb']
        ub_val = config['ub']
        for run in range(num_runs):
            # 使用不同的隨機種子，確保每次 Run 的搜索路徑不同
            seed = 100 + run 
            
            # 執行主程式並拿回數據
            history_best, final_best = run_esa_optimization(
                obj_func=obj_func, 
                lb_val=lb_val, 
                ub_val=ub_val,
                dim=dimension, 
                max_nfe=max_nfe, 
                seed=seed
            )
            
            all_runs_best_fitness.append(final_best)
            all_runs_history.append(history_best)
            
            print(f"Run {run+1:02d}/{num_runs:02d} 完成! 最終最優解: {final_best:.6e}")
            
        # 3. 數據處理：計算統計指標
        all_runs_best_fitness = np.array(all_runs_best_fitness)
        mean_val = np.mean(all_runs_best_fitness)
        std_val = np.std(all_runs_best_fitness)
        best_val = np.min(all_runs_best_fitness)
        worst_val = np.max(all_runs_best_fitness)
        
        summary_data.append({
            'Function': func_name,
            'Dim': dimension,
            'Best': best_val,
            'Worst': worst_val,
            'Mean': mean_val,
            'Std': std_val,
            'Optimum': FUNC_CONFIG[func_name]['f_opt']
        })
        
        # 4. 數據處理：將該函數的所有收斂曲線對齊，並計算「平均收斂曲線」
        # 由於演化步長不同可能導致歷史長度有些微差異，這裡將其轉換為標準二維陣列
        max_len = max(len(h) for h in all_runs_history)
        padded_history = []
        for h in all_runs_history:
            # 如果長度不足 max_nfe，用最後一個最優值補滿
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded_history.append(h[:max_nfe]) # 確保長度不超過上限
            
        padded_history = np.array(padded_history)
        mean_convergence = np.mean(padded_history, axis=0) # 對 20 次 Run 取平均
        
        # 保存單一函數的詳細收斂數據到 CSV
        df_conv = pd.DataFrame(padded_history).T
        df_conv.columns = [f'Run_{i+1}' for i in range(num_runs)]
        df_conv['Mean_Convergence'] = mean_convergence
        df_conv.to_csv(f'experiment_results/{func_name}_convergence.csv', index_label='NFE')
        
        # 5. 繪製單一函數的收斂圖
        plt.figure(figsize=(8, 5))
        plt.plot(mean_convergence, label=f'ESA Mean (Dim={dimension})', color='blue', linewidth=2)
        plt.title(f'Convergence Curve on {func_name.capitalize()}')
        plt.xlabel('Number of Function Evaluations (NFE)')
        plt.ylabel('Best Fitness Value (Log Scale)')
        plt.yscale('log') # 演化算法通常使用對數座標軸看收斂細節
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig(f'experiment_results/{func_name}_convergence_plot.png', dpi=300)
        plt.close()

    # 6. 將所有函數的統計總表匯出成一個最終的 CSV 報告
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('experiment_results/final_statistical_summary.csv', index=False)
    
    print("\n==================== 所有實驗完成！ ====================")
    print("產生的數據與圖表已儲存在 'experiment_results/' 資料夾中。")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    run_all_experiments()
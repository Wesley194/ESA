import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from esa_main_framework import run_esa_optimization
from esa_benchmark_functions import FUNC_CONFIG

# ==========================================
# 繪製多次實驗「平均收斂曲線」的專用函數
# ==========================================
def plot_runs_convergence(all_histories, func_name, dim, save_dir='multi_dim_results'):
    """
    繪製多次獨立實驗的收斂曲線，包含：
    1. 每次實驗的半透明歷史軌跡 (Individual Runs)
    2. 多次實驗的平均歷史軌跡 (Mean Convergence)
    """
    # 確保長度一致 (防呆)
    min_len = min(len(h) for h in all_histories)
    
    # 將每一條原始 y 軌跡，轉換為「截至目前的最佳值 (Running Best)」
    running_bests = [np.minimum.accumulate(h[:min_len]) for h in all_histories]
    
    # 計算所有 Runs 在每個 NFE 節點上的平均值
    avg_running_best = np.mean(running_bests, axis=0)
    
    plt.figure(figsize=(8, 5), dpi=120)
    
    # 1. 畫出 5 次個別的軌跡 (半透明淺藍色)
    for i, rb in enumerate(running_bests):
        label = 'Individual Runs (5)' if i == 0 else "" # 只讓圖例顯示一次
        plt.plot(range(1, min_len + 1), rb, color='#1f77b4', alpha=0.25, linewidth=1.2, label=label)
        
    # 2. 畫出平均軌跡 (粗紅色)
    plt.plot(range(1, min_len + 1), avg_running_best, color='#d62728', linewidth=2.5, label='Mean Convergence')
    
    # 設定 Y 軸為對數刻度
    plt.yscale('log')
    
    # 美化圖表
    plt.title(f"{func_name.capitalize()} ({dim}D) - Convergence over 5 Runs", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Function Evaluations (NFE)', fontsize=12)
    plt.ylabel('Best Fitness Value (Log Scale)', fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    # 自動存檔至指定資料夾
    filename = os.path.join(save_dir, f"{func_name}_{dim}D_convergence.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close() # 關閉畫布釋放記憶體
    print(f"📈 成功匯出收斂圖: '{filename}'")


def run_comprehensive_experiments():
    # 1. 實驗矩陣設定
    functions_to_test = ['ellipsoid', 'rosenbrock', 'griewank']
    dimensions_to_test = [30, 50, 100]  # 測試不同維度
    num_runs = 5                      # 每個組合重複 5 次
    max_nfe = 1000                     # 真實評估上限
    
    os.makedirs('multi_dim_results', exist_ok=True)
    all_combination_summary = []
    
    start_time = time.time()
    
    # 2. 開始巢狀迴圈：函數 -> 維度 -> 重複實驗
    for func_name in functions_to_test:
        for dim in dimensions_to_test:
            print(f"\n[測試中] 函數: {func_name.upper()} | 維度: {dim}D")
            
            run_results = []
            all_histories = [] # 用來搜集 5 次實驗的軌跡
            
            config = FUNC_CONFIG[func_name]
            obj_func = config['f']
            lb_val = config['lb']
            ub_val = config['ub']
            
            for run in range(num_runs):
                seed = 10 + run  # 固定隨機種子確保實驗可重複
                
                # 執行主程式
                final_best, history_y = run_esa_optimization(
                    obj_func=obj_func,
                    lb_val=lb_val,
                    ub_val=ub_val,
                    dim=dim,
                    max_nfe=max_nfe,
                    seed=seed
                )
                run_results.append(final_best)
                all_histories.append(history_y) # 儲存這一次 run 的歷史軌跡
            
            # 3. 該組合的 5 次實驗結束，立刻呼叫畫圖函數！
            plot_runs_convergence(all_histories, func_name, dim, save_dir='multi_dim_results')
            
            # 4. 計算該組合的統計指標
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

    # 5. 匯出最終表格
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
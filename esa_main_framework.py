"""
ESA主程式架構
需import不同的模組
"""
import numpy as np

#import 4個sampling actions
from esa_sampling_actions import a1_de_screening, a2_surrogate_local_search, a3_full_crossover, a4_trust_region
#import RBF模型
#import LHS抽樣

# RL Agent(Q-Learning)
class QAgent():
    """ 透過 RL(Q-learning) 決定要採取哪種策略 """
    def __init__(self):
        self.Q_table = np.full((8, 4), 0.25) 
        self.gamma = 0.9
        self.learning_rate = 0.1 # learning_rate(alpha)
        self.current_state = 0 # 初始狀態設為 0 (s1)

    def select_action(self):
        """ 實作 softmax (公式10)，並用輪盤法作出最終選擇 """
        # softmax
        Q_values = self.Q_table[self.current_state, :]
        exp_Q = np.exp(Q_values)
        probabilities = exp_Q / np.sum(exp_Q)

        # 輪盤法
        action = np.random.choice(a=[0, 1, 2, 3], p=probabilities)
        return action
    
    def update(self, action, reward, improved):
        """ 實作公式 6 """
        success = 1 if improved else 0
        next_state = 2 * action + success
        max_q_nex = np.max(self.Q_table[next_state, :])
        self.Q_table[self.current_state][action] = (1 - self.learning_rate) * self.Q_table[self.current_state][action] + self.learning_rate * (reward + self.gamma * max_q_nex)

        self.current_state = next_state

# 目標函數設定
def real_objective_function(x: np.ndarray) -> float:
    """
    填入論文中的Benchmark函數(如Ellipsoid, Rosenbrock)
    """

# ESA主迴圈
def run_esa_optimization():
    # 實驗參數設定
    dim = 30
    max_nfe = 1000
    lb = np.full(dim, -5.0)  # 填入正確下界
    ub = np.full(dim,  5.0)  # 填入正確上界
    
    rng = np.random.default_rng(42)
    q_agent = QAgent()
    
    # 資料庫初始化
    # 呼叫LHS取得初始點，並用real_objective_function算出初始適應值
    # initial_X = lhs_initialization(...)
    # initial_y = ...
    
    DB_X = np.empty((0, dim))  # 暫時代替
    DB_y = np.empty((0,))      # 暫時代替
    nfe = len(DB_X)
    
    # 記錄目前的歷史最佳解
    prev_best = np.min(DB_y) if len(DB_y) > 0 else np.inf
    
    # 主迴圈
    while nfe < max_nfe:
        
        # RL 選擇策略
        action_idx = q_agent.select_action()
        
        # 執行對應的 Sampling Action
        new_points = []
        if action_idx == 0:
            new_points = [a1_de_screening(DB_X, DB_y, real_objective_function, lb, ub, rng)]
        elif action_idx == 1:
            new_points = [a2_surrogate_local_search(DB_X, DB_y, real_objective_function, lb, ub, rng)]
        elif action_idx == 2:
            new_points = [a3_full_crossover(DB_X, DB_y, real_objective_function, lb, ub, rng)]
        elif action_idx == 3:
            new_points = a4_trust_region(DB_X, DB_y, real_objective_function, lb, ub, rng)
        
        if not new_points:
            continue
            
        # 更新資料庫與消耗的評估次數 (NFE)
        for x_new, f_new in new_points:
            if nfe >= max_nfe:
                break
            # 將 x_new 與 f_new 拼接到 DB_X 與 DB_y 中
            # DB_X = np.vstack([DB_X, x_new])
            # DB_y = np.append(DB_y, f_new)
            nfe += 1
            
        # 計算 Reward 並更新 RL Agent
        curr_best = np.min(DB_y) if len(DB_y) > 0 else np.inf
        improved = (curr_best < prev_best)
        
        # 進步則給予 reward=1，否則為 0
        reward = 1.0 if improved else 0.0
        
        # 呼叫 RL agent進行 Q-table 更新
        q_agent.update(action=action_idx, reward=reward, improved=improved)
        
        prev_best = curr_best
        # 可以在這裡加上 print 印出目前的 NFE 與 best fitness 方便追蹤進度
        

if __name__ == "__main__":
    # 執行主程式
    # run_esa_optimization()
    print("執行主程式")

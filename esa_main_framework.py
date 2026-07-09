"""
ESA主程式架構
需import不同的模組
"""
import numpy as np

#import 4個sampling actions
from esa_sampling_actions import a1_de_screening, a2_surrogate_local_search, a3_full_crossover, a4_trust_region
from RBF import RBFModel
from LHS import *
from q_agent import *
from dqn_agent import *
from esa_benchmark_functions import FUNC_CONFIG


# ESA主迴圈
def run_esa_optimization(agent_type="DQN", obj_func=None, lb_val=-5.0, ub_val=5.0, dim=30, max_nfe=1000, seed=42):
    
    lb = np.full(dim, lb_val)  
    ub = np.full(dim, ub_val)
    rng = np.random.default_rng(seed)

    if agent_type == "DQN":
        agent = DQNAgent(state_dim=10, action_dim=4)
    elif agent_type == "QL":
        agent = QAgent()
    else:
        raise ValueError("不支援的 Agent 類型！")
    
    # 資料庫初始化
    # 呼叫LHS取得初始點，並用objective_function算出初始適應值
    initial_samples = 100
    DB_X = LHS(sample=initial_samples, dimension=dim, lowerbound=lb[0], upperbound=ub[0])
    DB_y = np.array([obj_func(x) for x in DB_X])
    nfe = len(DB_X)
    
    prev_best = np.min(DB_y)
    
    # DQN 的 input
    recent_history = deque(maxlen=50)
    recent_history.append(0) 
    recent_rbf_error = 0.0 
    stagnation_counter = 0 
    prev_action = -1       # 記錄上一招
    last_a1_nfe = 100

    # 取得初始 State for DQN 
    state = None
    if agent_type == "DQN":
        state = get_current_state(DB_X, DB_y, nfe, max_nfe, recent_history, recent_rbf_error, stagnation_counter, prev_action, last_a1_nfe)
    
    loop_counter = 0

    # 主迴圈
    while nfe < max_nfe:
        
        # RL 選擇策略
        if agent_type == "DQN":
            action_idx = agent.select_action(state)
        else:
            # 原本的 QAgent 選擇策略不需要連續狀態
            action_idx = agent.select_action()
        
        # 執行對應的 Sampling Action
        new_points = []
        if action_idx == 0:
            new_points = [a1_de_screening(DB_X, DB_y, obj_func, lb, ub, rng)]
        elif action_idx == 1:
            new_points = [a2_surrogate_local_search(DB_X, DB_y, obj_func, lb, ub, rng)]
        elif action_idx == 2:
            new_points = [a3_full_crossover(DB_X, DB_y, obj_func, lb, ub, rng)]
        elif action_idx == 3:
            new_points = a4_trust_region(DB_X, DB_y, obj_func, lb, ub, rng)
        
        if not new_points:
            continue
            
        # 更新資料庫與消耗的評估次數 (NFE)
        for x_new, f_new in new_points:
            if nfe >= max_nfe:
                break
            # 將 x_new 與 f_new 拼接到 DB_X 與 DB_y 中
            DB_X = np.vstack([DB_X, x_new])
            DB_y = np.append(DB_y, f_new)
            nfe += 1
            

        curr_best = np.min(DB_y)
        improved = (curr_best < prev_best)
        done = (nfe >= max_nfe)

        if agent_type == "DQN":
            if action_idx == 0 and improved:
                last_a1_nfe = nfe
                
            # 2. 計算代理模型的近期相對誤差 (sMAPE)
            if len(DB_X) > 15:
                rbf_eval = RBFModel()
                rbf_eval.fit(DB_X[-51:-1], DB_y[-51:-1])
                
                f_pred = rbf_eval.predict_single(DB_X[-1])
                f_true = DB_y[-1]
                error = abs(f_pred - f_true) / (abs(f_pred) + abs(f_true) + 1e-8)
                recent_rbf_error = float(error)

            if improved:
                reward = 1.0               # 只要有進步就是 1 分 (不論多微小)
                stagnation_counter = 0     # 停滯歸零
                recent_history.append(1)
            else:
                reward = 0.0               # 沒進步就是 0 分 (不倒扣)
                stagnation_counter += 1    # 累積停滯步數 (給 DQN 觀察用)
                recent_history.append(0)
                
            done = (nfe >= max_nfe)
            
            # 取得 Next State (傳入剛出完的 action_idx 當作下一步的 prev_action)
            next_state = get_current_state(DB_X, DB_y, nfe, max_nfe, recent_history, recent_rbf_error, stagnation_counter, action_idx, last_a1_nfe)
            
            # 訓練 DQN
            agent.memory.push(state, action_idx, reward, next_state, done)
            agent.update()
            state = next_state

            if loop_counter % 10 == 0:
                agent.update_target_network()
            
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            
            print(f"NFE: {nfe:4d} | Best: {curr_best:.4e} | Skew: {state[3]:.2f} | Err: {recent_rbf_error:.2f} | Stag: {stagnation_counter:2d} | Action: a{action_idx+1}")
        

        # Q-learning
        elif agent_type == "QL":
            reward = 1.0 if improved else 0.0
            agent.update(action=action_idx, reward=reward, improved=improved)
            # 印出 Q-learning 簡單資訊
            print(f"NFE: {nfe:4d} | Best: {curr_best:.4e} | Action: a{action_idx+1}")
        
        prev_best = curr_best
        prev_action = action_idx
        loop_counter += 1
        
    return np.min(DB_y)
        

if __name__ == "__main__":

    target_function = 'rosenbrock' 
    
    config = FUNC_CONFIG[target_function]
    obj_func = config['f']
    lb_val = config['lb']
    ub_val = config['ub']
    
    print(f"測試函數: {target_function} | 邊界: [{lb_val}, {ub_val}]")
    
    # 執行主程式測試
    best_val = run_esa_optimization(
        agent_type="QL", 
        obj_func=obj_func, 
        lb_val=lb_val, 
        ub_val=ub_val,
        dim=30, 
        max_nfe=1000, 
        seed=42
    )
    
    print(f"\n {target_function} 最終最佳解: {best_val:.4e}")

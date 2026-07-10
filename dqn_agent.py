import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from scipy.stats import skew


def get_current_state(DB_X, DB_y, lb_val, ub_val, nfe, max_nfe, dim, recent_history, recent_rbf_error, stagnation_counter, prev_action):
    """
    dqn 需要的輸入整理，總共有 6 維特徵和 one-hot 過的上一次決策
    """
    # 預算消耗比例
    s_0 = nfe / max_nfe                        
    # 近期成功率
    s_1 = sum(recent_history) / len(recent_history) if len(recent_history) > 0 else 0.0 
    # 空間多樣性 for a2 and a4
    recent_window_s2 = min(len(DB_X), min(25 + dim, 60))
    if recent_window_s2 > 1:
        local_X = DB_X[-recent_window_s2:]
        std_per_dim = np.std(local_X, axis=0)
        range_per_dim = np.ptp(local_X, axis=0) # np.ptp 即 max - min
        range_per_dim = np.maximum(range_per_dim, 1e-8) # 防呆：避免除以 0
        s_2 = np.mean(std_per_dim / range_per_dim)
    else:
        s_2 = 0.0
    
    # Local Skewness
    s_3 = 0.0  
    local_window = 50
    
    if len(DB_X) > 100:
        local_y = DB_y[-local_window:]
        raw_skew = skew(local_y)
        if not np.isnan(raw_skew): 
            s_3 = float(np.tanh(raw_skew)) # 用 tanh 壓縮到安全範圍

    # RBF 模型誤差 (MAPE)
    s_4 = recent_rbf_error                     
    
    # 停滯指數 (超過 20 步沒進步就逼近 1.0)
    s_5 = min(stagnation_counter / 20.0, 1.0)
    
    # 組合 6 維基礎特徵
    base_state = np.array([s_0, s_1, s_2, s_3, s_4, s_5], dtype=np.float32)
    
    # 4 維 One-Hot Encoding
    action_one_hot = np.zeros(4, dtype=np.float32)
    if prev_action != -1:
        action_one_hot[prev_action] = 1.0
        
    return np.concatenate([base_state, action_one_hot])
    return np.concatenate([base_state])

class ReplayBuffer:
    """ 經驗回放池，用來打破資料的時間關聯性 """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """ DQN 神經網路結構 (MLP) """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    """ 透過 RL(DQN) 決定要採取哪種策略 """
    def __init__(self, state_dim=4, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 建立主網路與目標網路
        self.main_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval() # 目標網路不計算梯度
        
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(capacity=5000)
        
        self.batch_size = 32
        self.gamma = 0.9
        self.epsilon = 1.0          # 初始隨機探索機率
        self.epsilon_decay = 0.995  # 探索衰減率
        self.epsilon_min = 0.05     # 最低探索機率

    def select_action(self, state):
        """ 實作 Epsilon-greedy 策略 """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)
        return q_values.argmax().item()

    def update(self):
        """ 實作 DQN 的權重更新邏輯 """
        if len(self.memory) < self.batch_size:
            return

        # 從 Replay Buffer 抽樣
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # 計算目前 Q 值
        q_values = self.main_net(state).gather(1, action)

        # 計算目標 Q 值 (Bellman Equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * max_next_q

        # 計算 Loss 並更新 Main Net
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """ 定期同步目標網路的權重 """
        self.target_net.load_state_dict(self.main_net.state_dict())
import numpy as np

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
        self.Q_table[self.current_state, action] = (1 - self.learning_rate) * self.Q_table[self.current_state, action] + self.learning_rate * (reward + self.gamma * max_q_nex)

        self.current_state = next_state
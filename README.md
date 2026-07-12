# 論文實作與延伸：Evolutionary Sampling Agent (ESA) for Expensive Problems

本專案實作了論文 *Evolutionary Sampling Agent for Expensive Problems*，並在此基礎上導入 Deep Q-Network (DQN)，以探討在複雜的地形特徵下，Agent 策略選擇的泛化能力與收斂表現。

## Requirements

- **NumPy**
- **scipy**
- **pandas**
- **torch**
- **matplotlib**

---

## 核心實作與延伸

### 1. 論文重現
在此階段，我們完整重現了原論文的架構：
- **環境與策略**：實作了針對 Expensive Problems 的 4 種演化採樣策略。
- **決策機制**：使用 Q-Learning (Q-Table) 作為 Agent 的大腦，根據當前地形特徵選擇最適合的採樣策略。
- **加速**: @@@這邊記得要改，怎麼加速怎麼來@@@
### 2. 進階延伸 (Deep Q-Network)
論文中使用的 Q-Learning 僅有有限的 8 個狀態，但 Expensive Problems 中的複雜程度往往不僅於此，因此我們導入更加進階的模型DQN，試圖透過更進階的模型加快ESA的收斂速度。
#### 特徵定義
為了讓 MLP 能精確感知地形的變化進而選出更適合的策略，我們最終選出了 8 個特徵作為輸入：

1. **s_1 是否進步**：數值 0 或 1，表示前一次的採樣策略是否成功找到更佳解
2. **s_2 空間多樣性**：計算近期數據(設定數量與a2、a4的採樣數相近)在各維度的標準差與全距比例 (std / range)。此特徵用以評估當前搜索空間的多樣性與收斂狀態。
3. **s_3 Local Skewness**：計算近期目標函數值 (Fitness) 的偏度。使用 `tanh` 函數將其壓縮至 `[-1, 1]` 的安全範圍。此特徵用於反映局部地形的對稱性與崎嶇程度，同時避免極端值導致梯度爆炸。
4. **s_4 預測誤差**：代理模型 (RBF) 的預測誤差 (MAPE)，讓 Agent 知道當前地形是否已被模型充分掌握。
5. **Action History (4D One-Hot)**：將前次使用的策略進行 One-Hot 編碼，提供 Agent 決策上參考。

*(註：我們亦實作了消融機制(Ablation Mode)，用於分析各項地形特徵對決策成效的貢獻度。)*

#### 神經網路架構
底層採用 PyTorch 實作，網路架構如下：

- **神經網路 (MLP Structure)**：
  - **Input Layer**：接收上述的 8 維特徵。
  - **Hidden Layers**：包含兩層各 64 個節點的全連接層，hidden layers 中採用 `ReLU` 激勵函數。
  - **Output Layer**：輸出 4 維的 Q-value，對應 4 種演化採樣策略的預期回報。
- **RL 穩定機制**：
  - **Replay Buffer**：實作容量為 5000 的經驗回放池，透過隨機抽樣打破訓練資料的時間關聯性。
  - **Target Network**：採用雙網路架構，計算 Bellman Equation 目標值時切斷梯度，並定期同步權重，大幅降低訓練過程中的 Q 值震盪。
  - **Epsilon-Greedy 探索**：探索率 (Epsilon) 從 1.0 起始，以 0.995 的衰減率逐步收斂至最低 0.05，確保 Agent 在Exploration 與Exploitation 之間取得良好平衡。

---

## 實驗數據
### Q-Learning
| 論文數據 | 實驗結果 |
| :-----: | :--------: |
| <img width="293" height="242" alt="image" src="https://github.com/user-attachments/assets/fb3c5fad-45bb-481b-9a70-cce1f887a6c4" /> | <img width="480" height="266" alt="image" src="https://github.com/user-attachments/assets/053f02be-720a-4229-9a14-fe4e1e0d44f2" /> |
| <img width="870" height="243" alt="image" src="https://github.com/user-attachments/assets/4728bfc1-9c60-4759-beb6-1a800826a88b" /> |  |
| <img width="871" height="241" alt="image" src="https://github.com/user-attachments/assets/2941e93f-d130-4c4a-bf33-906d43cd6869" /> |  |

*像我貼的那篇一樣，左邊放論文的 table 235(沒做的記得省略)，右邊放我們跑出來的
### DQN
消融模式和table5
### Q-Learning 與 DQN 成效對照
這個數據我在跑，可以在看要不用放兩者的table5
---

## 總結與未來展望 (Conclusion)

- **總結**：
- **未來展望**：

## References
*   **[Baseline]** Zhen, H., Gong, W., & Wang, L. (2023). Evolutionary sampling agent for expensive problems. *IEEE Transactions on Evolutionary Computation*, 27(3), 716-727.
*   **[DQN Extension]** Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*.

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
- **代理模型**：引入 Radial Basis Function (RBF) 作為代理模型，透過先前數據建構代理模型來逼近真實的適應值評估 Fitness Evaluation。
- **採樣策略**：實作了針對 Expensive Problems 的 4 種演化採樣策略。
- **決策機制**：使用 Q-Learning (Q-Table) 作為 Agent 的大腦，根據當前地形特徵選擇最適合的採樣策略。

### 2. 進階延伸 (Deep Q-Network, DQN)
論文中使用的 Q-Learning 僅有有限的 8 個狀態，但 Expensive Problems 中的複雜程度往往不僅於此，因此我們導入DQN，試圖透過更進階的模型加快ESA的收斂速度。
#### 特徵定義
為了讓 DQN 的神經網路能精確感知地形的變化進而選出更適合的策略，我們最終選出了 8 個特徵作為輸入：

1. **s_1 是否進步**：數值 0 或 1，表示前一次的採樣策略是否成功找到更佳解
2. **s_2 空間多樣性**：計算近期數據 (設定數量與a2、a4的採樣數相近) 在各維度的標準差與全距比例 (std / range)。此特徵用以評估當前搜索空間的多樣性與收斂狀態。
3. **s_3 Local Skewness**：計算近期目標函數值 (Fitness) 的偏度。使用 `tanh` 函數將其壓縮至 `[-1, 1]` 的安全範圍。此特徵用於反映局部地形的對稱性與崎嶇程度，同時避免極端值導致梯度爆炸。
4. **s_4 預測誤差**：代理模型 (RBF) 的預測誤差 (MAPE)，讓 Agent 知道當前地形是否已被模型充分掌握。
5. **前次決策 (4D One-Hot)**：將前次使用的策略進行 One-Hot 編碼，提供 Agent 決策上參考。

*(註：我們亦實作了消融機制 Ablation Mode，用於分析各項地形特徵對決策成效的貢獻度。)*

#### 神經網路架構
底層採用 PyTorch 實作，網路架構如下：

- **神經網路 (MLP Structure)**：
  - **Input Layer**：接收上述的 8 維特徵。
  - **Hidden Layers**：包含兩層各 64 個節點的全連接層，hidden layers 中採用 `ReLU` 激勵函數。
  - **Output Layer**：輸出 4 維的 Q-value，對應 4 種演化採樣策略的預期回報。
- **RL 穩定機制**：
  - **Replay Buffer**：實作容量為 5000 的經驗回放池，透過隨機抽樣打破訓練資料的時間關聯性。
  - **Target Network**：採用雙網路架構，計算 Bellman Equation 目標值時切斷梯度，並定期同步權重，大幅降低訓練過程中的 Q 值震盪。
  - **Epsilon-Greedy 探索**：Epsilon 從 1.0 起始，以 0.995 的衰減率逐步收斂至最低 0.05，確保 Agent 在 Exploration 與 Exploitation 之間取得良好平衡。

---

## 實驗數據
### Q-Learning
| 論文數據 | 實驗結果 |
| :-----: | :--------: |
| Table2 <img width="870" height="243" alt="image" src="https://github.com/user-attachments/assets/4728bfc1-9c60-4759-beb6-1a800826a88b" /> | Table2 <img width="913" height="219" alt="image" src="https://github.com/user-attachments/assets/a2214797-c4cc-4d7e-a4d3-aa468f231ab9" /> |
| Table3 <img width="871" height="241" alt="image" src="https://github.com/user-attachments/assets/2941e93f-d130-4c4a-bf33-906d43cd6869" /> | Table3 <img width="1166" height="219" alt="image" src="https://github.com/user-attachments/assets/e075fc93-5d6a-4941-ac1f-c607d87901fa" /> |

ESA各種收斂圖
| <sup>函數</sup> ＼ <sub>維度</sub> | 30D | 50D | 100D |
| :--- | :---: | :---: | :---: |
| **Ellipsoid** | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/02ed39ac-c124-422a-aeee-52c48aa0139a" /> | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/badd72ad-f703-4643-81e9-988b354cf80d" /> | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/8a86ab21-bf2d-4901-9766-a549f6c29e94" /> |
| **Rosenbrock** | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/f90920d6-39dc-47eb-9b2d-9b590bf8fd5f" /> | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/3be13f6d-37c7-4fea-b72a-5c6ec0942aac" /> | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/8b9efac3-dfa0-4b44-8a5c-8da986d45adf" /> |
| **Griewank** | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/8dbe49a5-7b0c-4b42-93e8-6c478f07f04b" /> | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/26c40b15-f574-47ec-b8e5-3d398164d5d9" /> | <img width="947" height="587" alt="image" src="https://github.com/user-attachments/assets/ef6801d5-ccdd-4800-a26c-a88b256b77d0" /> |

單一採樣策略收斂圖
| <sup>函數</sup> ＼ <sub>維度</sub> | 30D | 50D | 100D |
| :--- | :---: | :---: | :---: |
| **Ellipsoid** | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/44f951af-1327-4ea3-a814-9b023ee3c3e5" /> | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/3818c716-f483-4cae-a4bd-b32c6553563f" /> | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/837392c6-f6b6-49f5-abfe-f7212edad1f3" /> |
| **Rosenbrock** | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/550cb37d-ca24-42ac-a916-bad3f3e4d137" /> | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/44134c43-3ca8-41ee-be14-080ac03943a0" /> | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/f626614a-bb93-4bfc-9308-babe171b3ee2" /> |
| **Griewank** | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/a76adcdc-7ba1-4c75-9822-f06b2ee55e48" /> | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/93c9bf73-c755-451e-a8b6-fa9f352c1a2a" /> | <img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/eafd6027-48f5-4720-98a3-0a8ff446d9c6" /> |

去一採樣策略收斂圖
| <sup>函數</sup> ＼ <sub>維度</sub> | 30D | 50D | 100D |
| :--- | :---: | :---: | :---: |
| **Ellipsoid** | <img width="960" height="600" alt="table3_ellipsoid_30D" src="https://github.com/user-attachments/assets/53110528-f5af-4e82-b876-4a22c85969f3" /> | <img width="960" height="600" alt="table3_ellipsoid_50D" src="https://github.com/user-attachments/assets/650b9063-589e-4e60-b5ad-71d3bf55ea6d" /> | <img width="960" height="600" alt="table3_ellipsoid_100D" src="https://github.com/user-attachments/assets/68c26022-02ce-4665-8d91-b74becc1a6a2" /> |
| **Rosenbrock** | <img width="960" height="600" alt="table3_rosenbrock_30D" src="https://github.com/user-attachments/assets/dc2675f8-d64d-42c6-a3c6-6df55a65a6e7" /> | <img width="960" height="600" alt="table3_griewank_50D" src="https://github.com/user-attachments/assets/a258cb0b-9106-4a4d-a4ea-c5ffb160f487" /> | <img width="960" height="600" alt="table3_rosenbrock_100D" src="https://github.com/user-attachments/assets/a2f25feb-94b7-4a92-91fb-31f0715045be" /> |
| **Griewank** | <img width="960" height="600" alt="table3_griewank_30D" src="https://github.com/user-attachments/assets/66892043-ad04-4d09-8c84-e4f66d62a4a9" /> | <img width="960" height="600" alt="table3_griewank_50D" src="https://github.com/user-attachments/assets/65af3125-62ae-4fd3-8dbf-f2a3909ab620" /> | <img width="960" height="600" alt="table3_rosenbrock_100D" src="https://github.com/user-attachments/assets/8888771f-9977-410b-a59f-5d9b4ab9d1d3" /> |

### DQN
#### DQN 消融模式之數據
<p align="center">
  <img src="./results/DQN_ablation.jpg" width="500">
</p>

#### DQN 收斂數據
<p align="center">
  <img src="./results/table5_DQN.jpg" width="400">
</p>

#### Q-Learning 與 DQN 收斂圖比較
| <sup>函數</sup> ＼ <sub>維度</sub> | 30D | 50D | 100D |
| :--- | :---: | :---: | :---: |
| **Ellipsoid** | <img src="./convergence_plots/conv_ellipsoid_dim30.png" width="100%" alt="Ellipsoid 30D" /> | <img src="./convergence_plots/conv_ellipsoid_dim50.png" width="100%" alt="Ellipsoid 50D" /> | <img src="./convergence_plots/conv_ellipsoid_dim100.png" width="100%" alt="Ellipsoid 100D" /> |
| **Rosenbrock** | <img src="./convergence_plots/conv_rosenbrock_dim30.png" width="100%" alt="Rosenbrock 30D" /> | <img src="./convergence_plots/conv_rosenbrock_dim50.png" width="100%" alt="Rosenbrock 50D" /> | <img src="./convergence_plots/conv_rosenbrock_dim100.png" width="100%" alt="Rosenbrock 100D" /> |
| **Griewank** | <img src="./convergence_plots/conv_griewank_dim30.png" width="100%" alt="Griewank 30D" /> | <img src="./convergence_plots/conv_griewank_dim50.png" width="100%" alt="Griewank 50D" /> | <img src="./convergence_plots/conv_griewank_dim100.png" width="100%" alt="Griewank 100D" /> |
---

## 總結與未來展望 (Conclusion)

- **總結**：
本專案重現了 ESA 演算法，更針對其效能與狀態表示的侷限性進行了改良。
  1. **突破狀態空間限制**：成功導入 DQN 取代傳統的 Q-Table，解決了原先演算法只能處理離散且低維度狀態的痛點。我們精心設計了 8 維連續地形特徵，提供 Agent 更細緻的地形感知能力。
  2. **嚴謹的成效驗證**：透過完整的單一策略分析、消融實驗以及 Q-Learning 與 DQN 的數據對照，我們證實了深度強化學習在面對複雜、未知且昂貴的目標函數時，能有效提升 Agent 策略選擇的泛化能力與最終收斂表現。
- **未來展望**：
基於目前的實作成果，我們認為本專案仍有一些可進一步探討的發展方向：
  1. **持續改良強化學習架構**：目前的決策大腦基於 Value-based 的 DQN 演算法。未來可繼續改良輸入特徵、引入其他架構(如 PPO)，或近一步改良成 multi-agent。期望能在連續且更複雜的地形特徵輸入下，進一步提升訓練的穩定性與收斂速度。
  2. **拓展至工程實務應用**：除了標準的測試函數，希望能將此框架應用於真實世界的高昂最佳化問題。
    這個我們用到一半，時間不夠啦...

## References
*   **[Baseline]** Zhen, H., Gong, W., & Wang, L. (2023). Evolutionary sampling agent for expensive problems. *IEEE Transactions on Evolutionary Computation*, 27(3), 716-727.
*   **[DQN Extension]** Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*.

# 論文實作與延伸：Evolutionary Sampling Agent (ESA) for Expensive Problems

本專案實作了論文 *Evolutionary Sampling Agent for Expensive Problems*，並在此基礎上導入 Deep Q-Network (DQN)，以探討在複雜的地形特徵下，Agent 策略選擇的泛化能力與收斂表現。

## Requirements

- **NumPy**
- **aaa**

---

## 核心實作與延伸

### 1. 論文重現
在此階段，我們完整重現了原論文的架構：
- **環境與策略**：實作了針對 Expensive Problems 的 4 種演化採樣策略。
- **決策機制**：使用傳統的 Q-Learning (Q-Table) 作為 Agent 的大腦，根據當前地形特徵選擇最適合的採樣策略。
- **加速**: @@@這邊記得要改，怎麼加速怎麼來@@@
### 2. 進階延伸 (Deep Q-Network)
論文中使用的 Q-Learning 僅有有限的 8 個 state，但 Expensive Problems


---

## 實驗數據
### Q-Learning
| 論文數據 | 實驗結果 |
| :-----: | :--------: |
| <img width="293" height="242" alt="image" src="https://github.com/user-attachments/assets/fb3c5fad-45bb-481b-9a70-cce1f887a6c4" /> | <img width="480" height="266" alt="image" src="https://github.com/user-attachments/assets/053f02be-720a-4229-9a14-fe4e1e0d44f2" /> |
| <img width="1443" height="269" alt="image" src="https://github.com/user-attachments/assets/7d7b758e-0c58-49b4-81cb-8ea3c64aa76d" />
 |  |

*像我貼的那篇一樣，左邊放論文的 table 235(沒做的記得省略)，右邊放我們跑出來的
### DQN
*特徵放了什麼，node和層數怎麼放，*
然後消融模式和table5
### Q-Learning 與 DQN 成效對照
兩個 table5 放過來
然後數據要在跑一個，畫在同一張圖
---

## 💡 總結與未來展望 (Conclusion)

- **總結**：
- **未來展望**：

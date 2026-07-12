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
Function,Dimension,Best,Worst,Mean,Std
ellipsoid,30D,3.3294e-01,2.1940e+01,7.1591e+00,8.0824e+00
ellipsoid,50D,1.8672e+02,7.7698e+02,4.1492e+02,2.0524e+02
ellipsoid,100D,1.6745e+04,2.9555e+04,2.1579e+04,4.4221e+03
rosenbrock,30D,2.7624e+01,1.1103e+03,4.4194e+02,4.5526e+02
rosenbrock,50D,4.4992e+03,3.1783e+04,1.3391e+04,9.5427e+03
rosenbrock,100D,2.3826e+06,4.8335e+06,3.9107e+06,9.1205e+05
griewank,30D,5.5423e-01,9.9748e-01,8.2589e-01,1.6499e-01
griewank,50D,2.0241e+01,4.9887e+01,3.7907e+01,1.2357e+01
griewank,100D,1.6382e+03,2.0771e+03,1.8827e+03,1.4122e+02
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

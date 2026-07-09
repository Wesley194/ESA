"""
這個檔案包含四種抽樣策略和三個輔助函數
這四個抽樣函數都採用一樣的介面設計：
def a(DB_X, DB_y, real_f, lb, ub, rng)
參數的意義：
DB_X (np.ndarray): 歷史資料庫中的自變數矩陣，大小為N乘d，N為已評估次數，d為維度。
DB_y (np.ndarray): 歷史資料庫中的目標函數值，是一個一維陣列，長度為N。
real_f (callable): 目標函數，接收 1D numpy array 作為輸入，回傳 float 數值。
lb (np.ndarray): 決策變數的下界，是一個一維陣列，長度為d。
ub (np.ndarray): 決策變數的上界，是一個一維陣列，長度為d。
rng (np.random.Generator): Numpy的亂數產生器。

輸出格式：
action_a1, a2, a3: 會回傳一個tuple(x_new, f_new)，x_new代表新找到的最佳解，f_new代表其真實評估值。
action_a4: 迭代 k_max 次，回傳的是一個串列 [(x1, f1), (x2, f2), ...]。
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")
from RBF import RBFModel

#輔助運算函數，負責代理模型的最佳化
#參考論文Page 720的Algorithm 3
def jade(surrogate, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    bounds = list(zip(lb.tolist(), ub.tolist()))
    
    def vectorized_predict(x_transposed):
        x_normal = x_transposed.T
        return surrogate.predict(x_normal)

    try:
        res = differential_evolution(
            vectorized_predict,
            bounds, 
            popsize=3,
            maxiter=15,
            tol=1e-3,
            vectorized=True,
            updating='deferred'
        )
        return np.clip(res.x, lb, ub)
    
    except Exception:
        return lb + (ub - lb) * 0.5

#輔助運算函數，負責邊界計算
#參考論文Page 720的Eq. 7 & Eq. 8
def local_range(X_local: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    margin = 0.5 * (X_local.max(axis=0) - X_local.min(axis=0))
    margin = np.maximum(margin, 1e-3)
    lb_l = np.maximum(X_local.min(axis=0) - margin, lb)
    ub_l = np.minimum(X_local.max(axis=0) + margin, ub)
    bad = lb_l >= ub_l
    lb_l[bad] = lb[bad]
    ub_l[bad] = ub[bad]
    return lb_l, ub_l

#輔助運算函數，負責DE篩選
#參考論文Page 720的Algorithm 2
def de(surrogate, population: np.ndarray,lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, d = population.shape
    trials = []
    for i in range(n):
        cands = [j for j in range(n) if j != i]
        r1, r2, r3 = rng.choice(cands, 3, replace=False)
        F_i = rng.uniform(0.4, 1.0)
        mutant = np.clip(population[r1] + F_i * (population[r2] - population[r3]), lb, ub)
        CR = rng.uniform(0.1, 1.0)
        j_rand = rng.integers(d)
        mask = rng.random(d) < CR
        mask[j_rand] = True
        trials.append(np.where(mask, mutant, population[i]))
    trials = np.array(trials)
    return trials[np.argmin(surrogate.predict(trials))].copy()

#4個Sampling Actions
#a1負責負責全域探索。取出資料庫中前n筆較佳資料作為群體，訓練Global RBF，接著使用 DE/rand/1 產生n個向量，透過Global RBF挑選其中預測值最佳的點進行評估。
#參考論文 Page 720 的 Algorithm 2
def a1_de_screening(DB_X: np.ndarray, DB_y: np.ndarray, real_f: callable,lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> tuple:
    N = len(DB_X)
    n = 50
    g = min(N, 300)
    sorted_idx = np.argsort(DB_y)
    P = DB_X[sorted_idx[:min(n, N)]].copy()
    rbf_g = RBFModel()
    rbf_g.fit(DB_X[sorted_idx[:g]], DB_y[sorted_idx[:g]])
    x_c = de(rbf_g, P, lb, ub, rng)
    return x_c, real_f(x_c)

#a2負責負責局部搜尋。取出資料庫中的l個最佳解，訓練Local RBF模型，限定在這些點所構成的範圍內，以JADE尋求極小值。
#參考論文 Page 720 的 Algorithm 3
def a2_surrogate_local_search(DB_X: np.ndarray, DB_y: np.ndarray, real_f: callable,lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> tuple:
    d = lb.shape[0]
    l = min(len(DB_X), min(25 + d, 60)) 
    sorted_idx = np.argsort(DB_y)
    X_loc = DB_X[sorted_idx[:l]]
    y_loc = DB_y[sorted_idx[:l]]

    #維持多項式系統所需的最少資料數
    if len(X_loc) <= d + 1:
        l2 = min(len(DB_X), d + 10)
        X_loc = DB_X[sorted_idx[:l2]]
        y_loc = DB_y[sorted_idx[:l2]]

    rbf_l = RBFModel()
    rbf_l.fit(X_loc, y_loc)
    lb_l, ub_l = local_range(X_loc, lb, ub)
    x_c = jade(rbf_l, lb_l, ub_l)
    return x_c, real_f(x_c)

#a3負責基因重組。找出前m筆資料，針對這m筆資料中的當前最佳解，逐一在各個維度上抽取這m筆資料的對應特徵進行替換，並用代理模型篩出最佳解。
#參考論文 Page 720 的 Algorithm 4
def a3_full_crossover(DB_X: np.ndarray, DB_y: np.ndarray, real_f: callable,lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> tuple:
    m = min(len(DB_X), 100)
    d = lb.shape[0]
    sorted_idx = np.argsort(DB_y)
    P = DB_X[sorted_idx[:m]].copy()
    y_P = DB_y[sorted_idx[:m]].copy()

    rbf_m = RBFModel()
    rbf_m.fit(P, y_P)
    x_best = P[0].copy()
    pi = rng.permutation(d)
    for i in pi:
        P_temp = np.tile(x_best, (m, 1))
        P_temp[:, i] = P[:, i]
        x_best = P_temp[np.argmin(rbf_m.predict(P_temp))].copy()

    x_c = np.clip(x_best, lb, ub)
    return x_c, real_f(x_c)

#a4負責局部微調。透過縮放Trust Region，每次迭代比較真實改善幅度與預測改善幅度的比例，動態放大或縮小搜索範圍。
#參考論文 Page 721 的 Algorithm 5
def a4_trust_region(DB_X: np.ndarray, DB_y: np.ndarray, real_f: callable,lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> list:
    m = min(len(DB_X), 100)
    k_max = 3
    d = lb.shape[0]
    sorted_idx = np.argsort(DB_y)
    DB_m = DB_X[sorted_idx[:m]].copy()
    y_m = DB_y[sorted_idx[:m]].copy()
    x_best = DB_m[0].copy()
    f_best = y_m[0]
    lb_l, ub_l = local_range(DB_m, lb, ub)

    #初始Trust Region半徑為m筆資料中適應值落差最大者之間距的一半
    min_pt = DB_m[np.argmin(y_m)]
    max_pt = DB_m[np.argmax(y_m)]
    delta = max(0.5 * np.linalg.norm(max_pt - min_pt), 1e-8)

    new_data = []
    for k in range(k_max):
        tr_lb = np.maximum(x_best - delta, lb_l)
        tr_ub = np.minimum(x_best + delta, ub_l)
        if np.any(tr_ub <= tr_lb):
            break

        #代理模型僅需訓練落在trust region範圍內的資料
        in_tr = np.all((DB_m >= tr_lb) & (DB_m <= tr_ub), axis=1)
        X_tr = DB_m[in_tr] if in_tr.sum() > d + 1 else DB_m
        y_tr = y_m[in_tr] if in_tr.sum() > d + 1 else y_m

        rbf_l = RBFModel()
        rbf_l.fit(X_tr, y_tr)

        #記錄x_best尚未更新前的代理模型預測值
        f_pred_xbest_before = rbf_l.predict_single(x_best)

        x_c = jade(rbf_l, tr_lb, tr_ub)
        f_c = real_f(x_c)
        new_data.append((x_c.copy(), f_c))

        f_pred_xc = rbf_l.predict_single(x_c)
        f_before  = f_best

        #更新最佳值
        if f_c < f_best:
            x_best = x_c.copy()
            f_best = f_c

        #Trust ratio λ: 若預測進步量太小則縮小trust region
        actual_impr = f_before - f_c
        surrogate_impr = f_pred_xbest_before - f_pred_xc
        lambda_k = actual_impr / surrogate_impr if abs(surrogate_impr) > 1e-30 else 0.5

        if lambda_k <= 0.25:
            delta = 0.25 * delta
        elif lambda_k >= 0.75:
            delta = 2.0  * delta
        delta = max(delta, 1e-12)

    return new_data

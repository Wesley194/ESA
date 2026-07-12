import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

class RBFModel:
    def __init__(self, kernel='gaussian'):    
        #kernel: 'gaussian'
        self.kernel = kernel
        self.X_train = None
        self.omega = None
        self.beta = None
        
        # 用於紀錄 Y 值的平均與標準差
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X, y):
        self.X_train = np.array(X)
        y_train = np.array(y).flatten()
        N, d = self.X_train.shape
        # 對目標值進行 Z-score 標準化
        self.y_mean = np.mean(y_train)
        self.y_std = np.std(y_train)
        
        # 避免所有 y 值都一樣導致除以 0
        if self.y_std < 1e-8:
            self.y_std = 1.0 
            
        y_norm = (y_train - self.y_mean) / self.y_std

        # 計算距離平方矩陣
        dist_matrix_sq = cdist(self.X_train, self.X_train, metric='sqeuclidean')

        if self.kernel == 'gaussian':
            # 依照論文設定 beta 參數
            D_max = np.sqrt(np.max(dist_matrix_sq))
            self.beta = D_max / ((d * N) ** (1 / d)) if D_max > 0 else 1.0
            if self.beta < 1e-8:
                self.beta = 1e-8
                
            # 高斯核函數
            Phi = np.exp(-dist_matrix_sq / (self.beta ** 2))
            
        elif self.kernel == 'cubic':
            # Cubic 三次核天然具備外推能力，不會衰減回 0
            Phi = (np.sqrt(dist_matrix_sq)) ** 3
        # 加入微小正則化 (Ridge Factor) 防止奇異矩陣
        # 當局部搜尋 a2, a4 找到極小的谷底時，點會擠在一起
        # 加上 1e-6 的單位矩陣，保證 np.linalg.solve 絕對不會報錯崩潰
        lambda_reg = 1e-6
        Phi_reg = Phi + np.eye(N) * lambda_reg
        
        # 求解權重
        self.omega = np.linalg.solve(Phi_reg, y_norm)

    def predict(self, X):
        X = np.array(X)
        
        # 若傳入一維陣列，自動轉為二維
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        dist_matrix_sq = cdist(X, self.X_train, metric='sqeuclidean')

        if self.kernel == 'gaussian':
            Phi_new = np.exp(-dist_matrix_sq / (self.beta ** 2))
        elif self.kernel == 'cubic':
            Phi_new = (np.sqrt(dist_matrix_sq)) ** 3
            
        # 預測出標準化後的數值
        y_pred_norm = np.dot(Phi_new, self.omega)
        
        # 還原真實數值
        y_pred = y_pred_norm * self.y_std + self.y_mean
        return y_pred.flatten()
        
    def predict_single(self, x):
        """兼容單一點預測的輔助函數"""
        return self.predict([x])[0]

#簡單測試區塊
if __name__ == "__main__":
    X_sample = np.array([[1.0, 2.0], [3.0, 5.0], [2.0, 1.0], [5.0, 4.0], [4.0, 2.0]])
    F_true = np.array([5.0, 34.0, 5.0, 41.0, 20.0])
    rbf = RBFModel()
    rbf.fit(X_sample, F_true)
    X_new = np.array([[2.5, 3.0], [4.5, 3.5]])
    print(f"預測結果: {rbf.predict(X_new)}")
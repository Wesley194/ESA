import numpy as np
class RBF :
    def __init__(self):
        self.X_train = None
        self.omega = None
        self.beta = None
    def fit(self, X ,Y):#訓練模型
        self.X_train = np.atleast_2d(X)
        Y = Y.reshape(-1, 1)
        N, d =self.X_train.shape
        X_sq = np.sum(self.X_train**2, axis = 1, keepdims = True)
        dist_matrix_sq = X_sq + X_sq.T - 2 * np.dot(self.X_train, self.X_train.T)
        dist_matrix_sq = np.maximum(dist_matrix_sq, 0)
        dist_matrix = np.sqrt(dist_matrix_sq)
        D_max = np.max(dist_matrix)
        self.beta = D_max * ((d * N) ** (-1.0 / d))
        if self.beta == 0:
            self.beta = 1e-6
        Phi = np.exp(-dist_matrix_sq / self.beta)
        self.omega = np.linalg.solve(Phi + 1e-8 * np.eye(N), Y)
    def predict(self, x):#輸出fitness結果
        x = np.atleast_2d(x)
        x_sq = np.sum(x**2, axis = 1, keepdims = True)
        x_train_sq = np.sum(self.X_train**2, axis = 1, keepdims = True).T
        dist_matrix_sq = x_sq + x_train_sq - 2 * np.dot(x, self.X_train.T)
        dist_matrix_sq = np.maximum(dist_matrix_sq, 0)
        Phi_new = np.exp(-dist_matrix_sq / self.beta)
        Y_hat = np.dot(Phi_new, self.omega)
        return Y_hat.flatten()

#test:
#X_sample = np.array([[1.0, 2.0], [3.0, 5.0], [2.0, 1.0], [5.0, 4.0], [4.0, 2.0]])
#F_true = np.array([12.5, 35.1, 8.2, 42.0, 21.4])
#rbf_s = RBF()
#rbf_s.fit(X_sample, F_true)
#X_new = np.array([[2.5, 3.0], [4.5, 3.5]])
#predict = rbf_s.predict(X_new)
#print(predict)
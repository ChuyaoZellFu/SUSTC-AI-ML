import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SMOSVM:
    def __init__(self, C=1.0, tol=1e-3, max_iter=100):
        self.C = C        # 惩罚参数
        self.tol = tol    # 容差
        self.max_iter = max_iter

    def _compute_kernel(self, X, i, j):
        # 线性核计算
        return np.dot(X[i], X[j])

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)  # 拉格朗日乘子
        self.b = 0                        # 偏置
        self.w = np.zeros(n_features)     # 权重向量（可通过alpha计算）

        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            for i in range(n_samples):
                # 计算预测值
                E_i = self._decision_function(X[i]) - y[i]
                
                # 检查KKT条件
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    # 随机选择另一个alpha
                    j = np.random.choice([x for x in range(n_samples) if x != i])
                    E_j = self._decision_function(X[j]) - y[j]

                    # 保存旧的alpha
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    # 计算L和H（alpha的上下界）
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    # 计算eta
                    eta = 2.0 * self._compute_kernel(X, i, j) - \
                          self._compute_kernel(X, i, i) - self._compute_kernel(X, j, j)
                    if eta >= 0:
                        continue

                    # 更新alpha[j]
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # 检查alpha[j]是否有变化
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新alpha[i]
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # 更新偏置b
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self._compute_kernel(X, i, i) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self._compute_kernel(X, i, j)
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * self._compute_kernel(X, i, j) - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self._compute_kernel(X, j, j)
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

            # 检查是否收敛
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break

        # 计算权重向量w
        self.w = np.dot((self.alpha * y).T, X)

    def _decision_function(self, X):
        return np.dot(self.w, X) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))

# 数据预处理
def preprocess_data(X, y):
    # 将标签转换为 +1 或 -1
    y = np.where(y == 0, -1, 1)
    return X, y

if __name__ == "__main__":
    # 加载数据
    data = np.loadtxt("Project_Final\processed_data.txt",delimiter=" ")
    X,y = data[:,:-1], data[:,-1]

    # 选择二分类任务，只使用类别0和1
    X = X[y != 2]
    y = y[y != 2]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # 训练线性SVM
    svm = SMOSVM(C=1.0, tol=1e-3, max_iter=100)
    svm.fit(X_train, y_train)

    # 测试并输出准确率
    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy (SMO Linear SVM): {accuracy * 100:.2f}%")

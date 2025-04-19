import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = None

    def fit(self, X, y, sample_weight):
        n_samples, n_features = X.shape
        best_error = np.inf
        best_feature_index = 0
        best_threshold = 0
        best_polarity = 1

        for feature_index in range(n_features):
            feature_values = np.sort(X[:, feature_index])
            thresholds = feature_values[1:]  # Exclude min value to prevent division by zero

            for threshold in thresholds:
                for polarity in [-1, 1]:
                    predictions = np.ones(n_samples)
                    predictions[X[:, feature_index] <= threshold] = -1 * polarity

                    error = np.sum(sample_weight[predictions != y])
                    if error < best_error:
                        best_error = error
                        best_feature_index = feature_index
                        best_threshold = threshold
                        best_polarity = polarity

        self.feature_index = best_feature_index
        self.threshold = best_threshold
        self.polarity = best_polarity

    def predict(self, X):
        predictions = np.ones(X.shape[0])
        predictions[X[:, self.feature_index] <= self.threshold] = -1 * self.polarity
        return predictions


class AdaBoost:
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.alpha = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        sample_weight = np.full((n_samples,self.n_estimators), 1 / n_samples)  # 初始化每个样本的权重为1/n_samples,行代表不同样本,列代表不同分类器
        self.estimators = [self.base_estimator() for _ in range(self.n_estimators)]

        for i in range(self.n_estimators):
            estimator = self.estimators[i]
            estimator.fit(X, y, sample_weight[:,i])#取第i个分类器，传入所有样本进行训练
            predictions = estimator.predict(X)#对应tn

            # 计算错误率
            errors = np.sum(sample_weight[:,i] * (predictions != y)) / np.sum(sample_weight[:,i])
            # 计算分类器的质量
            self.alpha.append(np.log((1.0 - errors) / (errors + 1e-10)))
            # 更新样本权重
            if(i<self.n_estimators-1):
                sample_weight[:,i+1] = sample_weight[:,i] * np.exp(-1/2*y*self.alpha[i]*predictions)
                # 归一化样本权重
                sample_weight[:,i+1] /= np.sum(sample_weight[:,i+1])

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators]).T
        weighted_sum = np.dot(predictions, self.alpha)
        return np.sign(weighted_sum)

def evaluate(y_test, y_pred):
    """
    Evaluate the model performance using accuracy, recall, precision, and F1 score.
    
    Parameters:
    y_test (numpy array): True labels.
    y_pred (numpy array): Predicted labels.
    
    Returns:
    tuple: Accuracy, Recall, Precision, and F1 score.
    """
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    A = (tp + tn) / len(y_test)
    P = tp / (tp + fp) if (tp + fp) > 0 else 0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0
    F_1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

    return A, R, P, F_1

# 使用AdaBoost
def main():
    data = np.loadtxt("Project_Final\\processed_data.txt", delimiter=" ")
    X, y = data[:, :-1], data[:, -1]
    y = y.astype(np.int64)
    mask = y != 2
    y = y[mask]
    X = X[mask, :]
    y[y == 1] = 0
    y[y == 3] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    y_train[y_train == 0] = -1
    np.savetxt('Project_Final\\X_train.txt', X_train, fmt='%f', delimiter=' ')
    np.savetxt('Project_Final\\y_train.txt', y_train, fmt='%f', delimiter=' ')

    A = []
    R = []
    P = []
    F_1 = []
    for i in range(1,30):
        # 创建AdaBoost实例
        ada_clf = AdaBoost(DecisionStump, n_estimators=i)

        # 训练AdaBoost分类器
        ada_clf.fit(X_train, y_train)

        # 预测
        y_pred = ada_clf.predict(X_test)
        y_pred.astype(np.int64)
        y_pred[y_pred == -1] = 0
        A0, R0, P0, F_10 = evaluate(y_test, y_pred)
        A.append(A0)
        R.append(R0)
        P.append(P0)
        F_1.append(F_10)
    
    print("Accuracy:", A[28])
    print("Recall:", R[28])
    print("Precision:", P[28])
    print("F1 Score:", F_1[28])
    
    # 设置图形的大小
    plt.figure(figsize=(14, 10))

    # 绘制准确率A随i的变化
    plt.subplot(2, 2, 1)
    plt.plot(range(1, 30), A, marker='o')
    plt.title('Accuracy vs. Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')

    # 绘制召回率R随i的变化
    plt.subplot(2, 2, 2)
    plt.plot(range(1, 30), R, marker='o')
    plt.title('Recall vs. Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')

    # 绘制精确度P随i的变化
    plt.subplot(2, 2, 3)
    plt.plot(range(1, 30), P, marker='o')
    plt.title('Precision vs. Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')

    # 绘制F1分数F_1随i的变化
    plt.subplot(2, 2, 4)
    plt.plot(range(1, 30), F_1, marker='o')
    plt.title('F1 Score vs. Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')

    # 调整子图间距
    plt.tight_layout()

    # 显示图形
    plt.show()
    

        

if __name__ == "__main__":
    main()

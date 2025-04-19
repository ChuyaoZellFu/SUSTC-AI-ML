import numpy as np 
from sklearn.metrics import accuracy_score 
from rff import NormalRFF
from solver import Solver
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class BiLinearSVC:
    r'''二分类线性SVM
    
    通过求解对偶问题

    .. math:: \min_{\alpha} \quad & \frac{1}{2} \alpha^T Q \alpha + p^T \alpha \\
                \text{s.t.} \quad & y^T \alpha = 0, \\
                                  & 0 \leq \alpha_i \leq C, i=1,\cdots,N

    得到决策边界

    .. math:: f(x) = \sum_{i=1}^N y_i \alpha_i  x_i^T x - \rho

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数,默认为1;
    max_iter : int, default=1000
        SMO算法迭代次数,默认1000;
    tol : float, default=1e-5
        SMO算法的容忍度参数,默认1e-5.
    '''
    def __init__(self,
                 C: float = 1.,
                 max_iter: int = 1000,
                 tol: float = 1e-5) -> None:
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.tol = tol 

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''训练模型

        Parameters
        ----------
        X : np.ndarray
            训练集特征;
        y : np.array
            训练集标签,建议0为负标签,1为正标签.
        '''
        X, y = np.array(X), np.array(y, dtype=float)
        y[y != 1] = -1
        N, self.n_features = X.shape
        p = -np.ones(N)

        w = np.zeros(self.n_features)
        Q = y.reshape(-1, 1) * y * np.matmul(X, X.T)
        solver = Solver(Q, p, y, self.C, self.tol)
        
        def func(i):
            return y * np.matmul(X, X[i]) * y[i]

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break

            delta_i, delta_j = solver.update(i, j, func)
            w += delta_i * y[i] * X[i] + delta_j * y[j] * X[j]
        else:
            print("LinearSVC not coverage with {} iterations".format(
                self.max_iter))

        self.coef_ = (w, solver.calculate_rho())
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        '''决策函数，输出预测值'''
        return np.matmul(self.coef_[0], np.array(X).T) - self.coef_[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''预测函数，输出预测标签(0-1)'''
        return (self.decision_function(np.array(X)) >= 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''评估函数，给定特征和标签，输出正确率'''
        return accuracy_score(y, self.predict(X))

 
class BiKernelSVC(BiLinearSVC):
    r'''二分类核SVM,优化问题与BiLinearSVC相同,只是Q矩阵定义不同。

    此时的决策边界

    .. math:: f(x) = \sum_{i=1}^N y_i \alpha_i K(x_i, x) - \rho

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数,默认为1;
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        核函数，默认径向基函数(RBF);
    degree : float, default=3
        多项式核的次数,默认3;
    gamma : {"scale", "auto", float}, default="scale"
        rbf、ploy和sigmoid核的参数 :math:`\gamma`，如果用'scale',那么就是1 / (n_features * X.var())，如果用'auto'，那么就是1 / n_features；
    coef0 : float, default=0.
        核函数中的独立项。它只在"poly"和"sigmoid"中有意义；
    max_iter : int, default=1000
        SMO算法迭代次数,默认1000;
    rff : bool, default=False
        是否采用随机傅里叶特征,默认为False;
    D : int, default=1000
        随机傅里叶特征的采样次数,默认为1000;
    tol : float, default=1e-5
        SMO算法的容忍度参数,默认1e-5.
    '''
    def __init__(self,
                 C: float = 1.,
                 kernel: str = 'rbf',
                 degree: float = 3,
                 gamma: str = 'scale',
                 coef0: float = 0,
                 max_iter: int = 1000,
                 rff: bool = False,
                 D: int = 1000,
                 tol: float = 1e-5) -> None:
        super().__init__(C, max_iter, tol)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.D = D

    def register_kernel(self, std: float):
        '''注册核函数
        
        Parameters
        ----------
        std : 输入数据的标准差,用于rbf='scale'的情况
        '''
        if type(self.gamma) == str:
            gamma = {
                'scale': 1 / (self.n_features * std),
                'auto': 1 / self.n_features,
            }[self.gamma]
        else:
            gamma = self.gamma

        if self.rff:
            rff = NormalRFF(gamma, self.D).fit(np.ones((1, self.n_features)))
            rbf_func = lambda x, y: np.matmul(rff.transform(x),
                                              rff.transform(y).T)
        else:
            rbf_func = lambda x, y: np.exp(-gamma * (
                (x**2).sum(1, keepdims=True) +
                (y**2).sum(1) - 2 * np.matmul(x, y.T)))

        degree = self.degree
        coef0 = self.coef0
        return {
            "linear": lambda x, y: np.matmul(x, y.T),
            "poly": lambda x, y: (gamma * np.matmul(x, y.T) + coef0)**degree,
            "rbf": rbf_func,
            "sigmoid": lambda x, y: np.tanh(gamma * np.matmul(x, y.T) + coef0)
        }[self.kernel]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.array(X), np.array(y, dtype=float)
        y[y != 1] = -1
        N, self.n_features = X.shape
        p = -np.ones(N)

        kernel_func = self.register_kernel(X.std())

        Q = y.reshape(-1, 1) * y * kernel_func(X, X)
        solver = Solver(Q, p, y, self.C, self.tol)
        

        def func(i):
            return y * kernel_func(X, X[i:i + 1]).flatten() * y[i]

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break
            solver.update(i, j, func)
        else:
            print("KernelSVC not coverage with {} iterations".format(
                self.max_iter))

        self.decision_function = lambda x: np.matmul(
            solver.alpha * y,
            kernel_func(X, x),
        ) - solver.calculate_rho()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return super().score(X, y)

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


if __name__ == "__main__":
    data = np.loadtxt("Project_Final\processed_data.txt",delimiter=" ")
    X,y = data[:,:-1], data[:,-1]
    y = y.astype(np.int64)
    mask = y!=2
    y = y[mask]
    X = X[mask,:]
    y[y==1] = 0
    y[y==3] = 1
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    y_train[y_train==0] = -1
    np.savetxt('Project_Final\X_train.txt', X_train, fmt='%f', delimiter=' ')
    np.savetxt('Project_Final\y_train.txt', y_train, fmt='%f', delimiter=' ')
    
    clf_linear = BiLinearSVC(C=1, max_iter=1000, tol=1e-5)
    clf_linear.fit(X_train,y_train)
    y_pred = clf_linear.predict(X_test)
    print(f"The y_pred of Linear SVM is {y_pred}")
    print(f"The y_test of Linear SVM is {y_test}")
    A, R, P, F_1 = evaluate(y_test, y_pred)
    print(f"The A of Linear SVM is {A}")
    print(f"The R of Linear SVM is {R}")
    print(f"The P of Linear SVM is {P}")
    print(f"The F_1 of Linear SVM is {F_1}")

    clf_kernel = BiKernelSVC(C=1, kernel='rbf', max_iter=1000, tol=1e-5)
    clf_kernel.fit(X_train,y_train)
    y_pred = clf_kernel.predict(X_test)
    print(f"The y_pred of Kernel SVM is {y_pred}")
    print(f"The y_test of Kernel SVM is {y_test}")
    A_kernel, R_kernel, P_kernel, F_1_kernel = evaluate(y_test, y_pred)
    print(f"The A of Kernel SVM is {A_kernel}")
    print(f"The R of Kernel SVM is {R_kernel}")
    print(f"The P of Kernel SVM is {P_kernel}")
    print(f"The F_1 of Kernel SVM is {F_1_kernel}")

        # 设置图形的大小
    plt.figure(figsize=(10, 8))

    # 定义条形图的位置和宽度
    bar_width = 0.35  # 条形图的宽度
    index = np.arange(4)  # 四个性能指标

    # 绘制Linear SVM的条形图
    linear_bars = plt.bar(index, [A, R, P, F_1], bar_width, label='Linear SVM')

    # 绘制Kernel SVM的条形图
    kernel_bars = plt.bar(index + bar_width, [A_kernel, R_kernel, P_kernel, F_1_kernel], bar_width, label='Kernel SVM')

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title('Performance Comparison between Linear SVM and Kernel SVM')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')

    # 设置x轴的刻度标签
    plt.xticks(index + bar_width / 2, ['Accuracy', 'Recall', 'Precision', 'F1 Score'])

    # 在条形图上方添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(linear_bars)
    add_labels(kernel_bars)

    # 显示图形
    plt.show()
    
    

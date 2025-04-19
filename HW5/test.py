import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 生成数据集
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.05, random_state=42,class_sep=0.3)
y = np.array(y)
print(y)
print(y[1])
print(X[1])

# 添加非线性特征
#rng = np.random.RandomState(42)
X[:, 0] = np.sin(X[:, 0])  # 使用正弦函数转换第一个特征
#X[:, 1] = np.cos(X[:, 1])  # 使用余弦函数转换第二个特征

# 可视化生成的数据集
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
plt.title('Non-linearly Separable Dataset')
plt.xlabel('Transformed Feature 1')
plt.ylabel('Transformed Feature 2')
plt.show()


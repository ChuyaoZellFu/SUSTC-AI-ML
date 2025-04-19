import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from matplotlib.colors import ListedColormap

def soft_k_means(X, k, max_iters=100, beta=1.0):
    # X: 数据点，形状为 (n_samples, n_features)
    # k: 聚类数量
    # max_iters: 最大迭代次数
    # beta: 高斯分布的温度参数，控制簇的软聚类程度

    # 随机初始化质心
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for i in range(max_iters):
        # 1. 计算每个点到每个质心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # 2. 计算隶属度 (基于高斯分布的概率)
        exp_distances = np.exp(-beta * distances)
        membership = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
        
        # 3. 更新质心
        new_centroids = np.dot(membership.T, X) / np.sum(membership, axis=0)[:, np.newaxis]
        
        # 如果质心没有变化，结束迭代
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return centroids, membership


if __name__ == '__main__':

    data = np.loadtxt("Project_Final\processed_data.txt", delimiter=" ")
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    k = 3
    centroids,labels = soft_k_means(X, k ,max_iters=100)
    
    centroids, membership = soft_k_means(X, k)

    # 为每个簇分配真实标签
    cluster_labels = np.zeros(k)
    for i in range(k):
        cluster_data = y[np.argmax(membership, axis=1) == i]  # 属于簇 i 的真实标签
        cluster_labels[i] = mode(cluster_data)[0]  # 使用众数作为簇的标签

    # 将Soft K-Means的聚类标签映射到真实标签
    y_pred = np.array([cluster_labels[np.argmax(membership[i])] for i in range(X.shape[0])])

    # 计算准确率
    accuracy = accuracy_score(y, y_pred)
    print(f"Soft K-Means Accuracy: {accuracy * 100:.2f}%")

   # 可视化
    plt.figure(figsize=(8, 6))

    # 使用不同的颜色深度表示数据点对簇的隶属度
    cmap = ListedColormap(plt.cm.get_cmap("tab10", k).colors)

    # 绘制数据点，使用透明度根据隶属度的大小绘制
    for i in range(k):
        plt.scatter(X[:, 0], X[:, 1], c=membership[:, i], cmap=cmap, alpha=0.7, label=f'Cluster {i+1}')
        
    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')

    # 添加标题和图例
    plt.title("Soft K-Means Clustering on Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 显示图形
    plt.colorbar(label="Membership Probability")
    plt.show()

    data = np.loadtxt("Project_Final\X_std_encode_dim2.txt", delimiter=" ")
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    k = 3
    centroids,labels = soft_k_means(X, k ,max_iters=100)
    
    centroids, membership = soft_k_means(X, k)

    # 为每个簇分配真实标签
    cluster_labels = np.zeros(k)
    for i in range(k):
        cluster_data = y[np.argmax(membership, axis=1) == i]  # 属于簇 i 的真实标签
        cluster_labels[i] = mode(cluster_data)[0]  # 使用众数作为簇的标签

    # 将Soft K-Means的聚类标签映射到真实标签
    y_pred = np.array([cluster_labels[np.argmax(membership[i])] for i in range(X.shape[0])])

    # 计算准确率
    accuracy = accuracy_score(y, y_pred)
    print(f"Soft K-Means Accuracy: {accuracy * 100:.2f}%")

   # 可视化
    plt.figure(figsize=(8, 6))

    # 使用不同的颜色深度表示数据点对簇的隶属度
    cmap = ListedColormap(plt.cm.get_cmap("tab10", k).colors)

    # 绘制数据点，使用透明度根据隶属度的大小绘制
    for i in range(k):
        plt.scatter(X[:, 0], X[:, 1], c=membership[:, i], cmap=cmap, alpha=0.7, label=f'Cluster {i+1}')
        
    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')

    # 添加标题和图例
    plt.title("Soft K-Means Clustering on Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 显示图形
    plt.colorbar(label="Membership Probability")
    plt.show()

    data = np.loadtxt("Project_Final\X_std_PCA_dim2.txt", delimiter=" ")
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    k = 3
    centroids,labels = soft_k_means(X, k ,max_iters=100)
    
    centroids, membership = soft_k_means(X, k)

    # 为每个簇分配真实标签
    cluster_labels = np.zeros(k)
    for i in range(k):
        cluster_data = y[np.argmax(membership, axis=1) == i]  # 属于簇 i 的真实标签
        cluster_labels[i] = mode(cluster_data)[0]  # 使用众数作为簇的标签

    # 将Soft K-Means的聚类标签映射到真实标签
    y_pred = np.array([cluster_labels[np.argmax(membership[i])] for i in range(X.shape[0])])

    # 计算准确率
    accuracy = accuracy_score(y, y_pred)
    print(f"Soft K-Means Accuracy: {accuracy * 100:.2f}%")

   # 可视化
    plt.figure(figsize=(8, 6))

    # 使用不同的颜色深度表示数据点对簇的隶属度
    cmap = ListedColormap(plt.cm.get_cmap("tab10", k).colors)

    # 绘制数据点，使用透明度根据隶属度的大小绘制
    for i in range(k):
        plt.scatter(X[:, 0], X[:, 1], c=membership[:, i], cmap=cmap, alpha=0.7, label=f'Cluster {i+1}')
        
    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')

    # 添加标题和图例
    plt.title("Soft K-Means Clustering on PCA Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 显示图形
    plt.colorbar(label="Membership Probability")
    plt.show()
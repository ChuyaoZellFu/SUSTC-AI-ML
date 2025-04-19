import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import mode

def k_means_plus_plus(X, k, max_iters=100):
    # X: 输入数据，形状为 (n_samples, n_features)
    # k: 聚类簇数
    # max_iters: 最大迭代次数

    # 第一步：随机选择一个数据点作为第一个质心
    centroids = [X[np.random.choice(X.shape[0])]]

    for _ in range(1, k):
        # 计算每个点到最近质心的距离的平方
        distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)**2
        
        # 通过距离的平方来选择下一个质心的概率
        prob = distances / distances.sum()

        # 根据概率分布选择下一个质心
        next_centroid = X[np.random.choice(X.shape[0], p=prob)]
        centroids.append(next_centroid)

    centroids = np.array(centroids)
    
    # 转为整数索引
    centroids = centroids.astype(int)

    for i in range(max_iters):
        # 1. 分配数据点到最近的质心
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 2. 更新质心
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 如果质心没有变化，结束迭代
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels


if __name__ == '__main__':

    data = np.loadtxt("Project_Final\processed_data.txt", delimiter=" ")
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    k = 3
    centroids,labels = k_means_plus_plus(X, k ,max_iters=100)
    
    # 计算每个簇的多数标签
    cluster_labels = np.zeros(k)
    for i in range(k):
        cluster_data = y[labels == i]  # 当前簇的真实标签
        cluster_labels[i] = mode(cluster_data)[0]  # 使用众数作为簇的标签

    # 将K-Means的聚类标签映射到真实标签
    y_pred = np.array([cluster_labels[label] for label in labels])

    # 计算准确率
    accuracy = accuracy_score(y, y_pred)
    print(f"K-Means Accuracy: {accuracy * 100:.2f}%")

    plt.figure(figsize=(8, 6))

    # 绘制所有数据点
    for i in range(k):
        plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f'Cluster {i+1}')

    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')

    # 添加标题和图例
    plt.title("K-Means++ Clustering on Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 显示图形
    plt.show()

    data = np.loadtxt("Project_Final\X_std_encode_dim2.txt", delimiter=" ")
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    k = 3
    centroids,labels = k_means_plus_plus(X, k ,max_iters=100)
    
    # 计算每个簇的多数标签
    cluster_labels = np.zeros(k)
    for i in range(k):
        cluster_data = y[labels == i]  # 当前簇的真实标签
        cluster_labels[i] = mode(cluster_data)[0]  # 使用众数作为簇的标签

    # 将K-Means的聚类标签映射到真实标签
    y_pred = np.array([cluster_labels[label] for label in labels])

    # 计算准确率
    accuracy = accuracy_score(y, y_pred)
    print(f"K-Means Accuracy: {accuracy * 100:.2f}%")

    plt.figure(figsize=(8, 6))

    # 绘制所有数据点
    for i in range(k):
        plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f'Cluster {i+1}')

    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')

    # 添加标题和图例
    plt.title("K-Means++ Clustering on Encoded Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 显示图形
    plt.show()

    data = np.loadtxt("Project_Final\X_std_PCA_dim2.txt", delimiter=" ")
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    k = 3
    centroids,labels = k_means_plus_plus(X, k ,max_iters=100)
    
    # 计算每个簇的多数标签
    cluster_labels = np.zeros(k)
    for i in range(k):
        cluster_data = y[labels == i]  # 当前簇的真实标签
        cluster_labels[i] = mode(cluster_data)[0]  # 使用众数作为簇的标签

    # 将K-Means的聚类标签映射到真实标签
    y_pred = np.array([cluster_labels[label] for label in labels])

    # 计算准确率
    accuracy = accuracy_score(y, y_pred)
    print(f"K-Means Accuracy: {accuracy * 100:.2f}%")

    plt.figure(figsize=(8, 6))

    # 绘制所有数据点
    for i in range(k):
        plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f'Cluster {i+1}')

    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')

    # 添加标题和图例
    plt.title("K-Means Clustering on PCA Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 显示图形
    plt.show()
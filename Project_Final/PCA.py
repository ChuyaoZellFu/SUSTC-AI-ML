import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    data = np.loadtxt("Project_Final\processed_data.txt",delimiter=" ")
    X,y = data[:,:-1], data[:,-1]

    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    cov_mat = np.cov(X_std.T)
    eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)

    eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda k :k[0], reverse=True)

    #dim = 2
    w_2 = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
    print('Matrix W in dim2:\n',w_2)

    X_train_0_pca = X_std[0].dot(w_2)
    print(X_train_0_pca)
    X_train_pca_2 = X_std.dot(w_2)
    X_train_pca_2 = np.append(X_train_pca_2, y[:, np.newaxis], axis=1)
    np.savetxt('Project_Final\X_std_PCA_dim2.txt', X_train_pca_2, fmt='%f', delimiter=' ')

    plt.figure()
    colors = ['r','b','g']
    markers = ['s','x','o']

    plt.title("Projection of Original Data on the first two principal components")
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X_std[y==l, 0], X_std[y==l, 1], c=c, label=l, marker=m)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

    #dim = 3
    w_3 = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis],eigen_pairs[2][1][:,np.newaxis]))
    print('Matrix W in dim3:\n',w_3)

    X_train_0_pca = X_std[0].dot(w_3)
    print(X_train_0_pca)
    X_train_pca_3 = X_std.dot(w_3)
    X_train_pca_3 = np.append(X_train_pca_3, y[:, np.newaxis], axis=1)
    np.savetxt('Project_Final\X_std_PCA_dim3.txt', X_train_pca_3, fmt='%f', delimiter=' ')

    colors = ['r','b','g']
    markers = ['s','x','o']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 创建3D坐标轴

    for l, c, m in zip(np.unique(y), colors, markers):
        # 选择当前类别的数据
        class_data = X_std[y == l]
        # 绘制散点图
        ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], c=c, label=l, marker=m)

    ax.set_title("Projection of Original Data on the first three principal components")
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

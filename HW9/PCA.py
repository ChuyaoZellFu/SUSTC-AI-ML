import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    df_wine = pd.read_csv('HW9\wine.data', header=None)
    X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    cov_mat = np.cov(X_train_std.T)
    eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)

    eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda k :k[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
    print('Matrix W:\n',w)
    for i in range(2,len(eigen_pairs)):
        u = np.hstack((w,eigen_pairs[i][1][:,np.newaxis]))

    X_train_0_pca = X_train_std[0].dot(w)
    print(X_train_0_pca)
    X_train_pca = X_train_std.dot(w)

    X_reconstructed = X_train_pca.dot(w.T) + np.mean(X_train_std,axis=0).dot(u).dot(u.T)
    X_reconstructed = sc.inverse_transform(X_reconstructed)
    mse = np.mean((X_train_std - X_reconstructed)**2)
    print(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(10, 5))

    colors = ['r','b','g']
    markers = ['s','x','o']

    # 原始数据
    plt.subplot(1, 2, 1)
    plt.title("Original Data")
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_std[y_train==l, 0], X_train_std[y_train==l, 1], c=c, label=l, marker=m)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='lower left')

    # 重建后的数据
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Data")
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_reconstructed[y_train==l, 0], X_reconstructed[y_train==l, 1], c=c, label=l, marker=m)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()



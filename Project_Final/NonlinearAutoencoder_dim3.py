import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Linear layer definition
class Linear:
    def __init__(self, input_dim, output_dim, bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        if bias:
            self.b = np.random.randn(output_dim) * 0.01
        else:
            self.b = np.zeros(output_dim) * 0.01

    def forward(self, input):
        return np.matmul(input, self.W) + self.b

    def backward(self, input, grad_output):
        grad_input = np.matmul(grad_output, self.W.T)
        grad_W = np.matmul(input.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)
        return grad_input, grad_W, grad_b

# ReLU activation function
class ReLU:
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(0, input)  # ReLU activation

    def backward(self, input, grad_output):
        grad_input = grad_output * (input > 0)  # Gradient of ReLU
        return grad_input

# MSE loss function
class MSE_Loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]

# Nonlinear Autoencoder definition
class Autoencoder:
    def __init__(self, n_feature=1, n_iter=200, lr=1e-4, tol=None, momentum=0.9):
        self.encoder = Linear(n_feature, 3, bias=True)  # 2 principal components
        self.decoder = Linear(3, n_feature, bias=True)
        self.act = ReLU()  # ReLU activation function
        self.loss_fn = MSE_Loss()
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.loss = []
        self.best_loss = np.inf
        self.patience = 20
        self.momentum = momentum
        self.v_W_enc = np.zeros_like(self.encoder.W)
        self.v_b_enc = np.zeros_like(self.encoder.b)
        self.v_W_dec = np.zeros_like(self.decoder.W)
        self.v_b_dec = np.zeros_like(self.decoder.b)

    def forward(self, X):
        encoded = self.encoder.forward(X)
        encoded = self.act.forward(encoded)  # Apply ReLU after encoder
        decoded = self.decoder.forward(encoded)
        return encoded, decoded

    def batch_update(self, X):
        epoch_not_improve = 0
        for iter in range(self.n_iter):
            encoded, decoded = self.forward(X)
            loss = self.loss_fn.forward(decoded, X)
            self.loss.append(loss)

            if self.tol is not None:
                if loss < self.best_loss - self.tol:
                    self.best_loss = loss
                    epoch_not_improve = 0
                elif np.abs(self.best_loss - loss) < self.tol:
                    epoch_not_improve += 1
                if epoch_not_improve > self.patience:
                    print(f"Stopping early at iteration {iter} with loss: {loss}")
                    break

            grad_output = self.loss_fn.backward(decoded, X)
            grad_encoded, grad_W_dec, grad_b_dec = self.decoder.backward(encoded, grad_output)
            grad_encoded = self.act.backward(encoded, grad_encoded)  # Backprop through ReLU
            _, grad_W_enc, grad_b_enc = self.encoder.backward(X, grad_encoded)

            self.v_W_enc = self.momentum * self.v_W_enc + (1 - self.momentum) * grad_W_enc
            self.v_b_enc = self.momentum * self.v_b_enc + (1 - self.momentum) * grad_b_enc
            self.v_W_dec = self.momentum * self.v_W_dec + (1 - self.momentum) * grad_W_dec
            self.v_b_dec = self.momentum * self.v_b_dec + (1 - self.momentum) * grad_b_dec

            self.encoder.W -= self.lr * self.v_W_enc
            self.encoder.b -= self.lr * self.v_b_enc
            self.decoder.W -= self.lr * self.v_W_dec
            self.decoder.b -= self.lr * self.v_b_dec

    def train(self, X_train):
        self.batch_update(X_train)

    def reconstruct(self, X):
        _, decoded = self.forward(X)
        return decoded
    
    def encoded(self, X):
        encoded, _ = self.forward(X)
        return encoded

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid()
        plt.show()

if __name__ == '__main__':
    data = np.loadtxt("Project_Final\processed_data.txt",delimiter=" ")
    X,y = data[:,:-1], data[:,-1]

    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    _, n_feature = X_std.shape
    model = Autoencoder(n_feature=n_feature, n_iter=7000, lr=5e-3, tol=1e-5)
    model.train(X_std)
    X_encoded = model.encoded(X_std)
    X_encoded = np.append(X_encoded, y[:, np.newaxis], axis=1)
    np.savetxt('Project_Final\X_std_encode_dim3.txt', X_encoded, fmt='%f', delimiter=' ')

    # Plot loss curve
    model.plot_loss()

    # Plot original and reconstructed data
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

class Identity:
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        return grad_output

class MSE_Loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]

class Autoencoder:
    def __init__(self, n_feature=1, n_iter=200, lr=1e-4, tol=None, momentum=0.9):
        self.encoder = Linear(n_feature, 2, bias=True)  # 2 principal components
        self.decoder = Linear(2, n_feature, bias=True)
        self.act = Identity()
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

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid()
        plt.show()

if __name__ == '__main__':
    df_wine = pd.read_csv('HW9\wine.data', header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    _, n_feature = X_train_std.shape
    model = Autoencoder(n_feature=n_feature, n_iter=3000, lr=3e-3, tol=1e-5)
    model.train(X_train_std)

    X_reconstructed = model.reconstruct(X_test_std)

    # Plot loss curve
    model.plot_loss()

    # Plot original and reconstructed data
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Data")
    plt.scatter(X_test_std[:, 0], X_test_std[:, 1], c=y_test, cmap=plt.cm.Paired, marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Data")
    plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c=y_test, cmap=plt.cm.Paired, marker='x')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

    loss = model.loss_fn.forward(X_reconstructed, X_test_std)
    print(f"Mean Squared Error: {loss}")

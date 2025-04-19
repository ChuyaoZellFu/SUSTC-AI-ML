import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Linear:
    def __init__(self, input_dim, output_dim, bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim)*0.01

        if bias:
            self.b = np.random.randn(output_dim)*0.01
        else:
            self.b = np.zeros(output_dim)*0.01

    def forward(self, input):
        """
        Forward propagation: Z = XW + b
        """
        output = np.matmul(input, self.W) + self.b
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            print(self.W)
            print("Weights contain NaN or Inf")
        return output

    def backward(self, input, grad_output):
        """
        Backward propagation: Compute gradients for W, b, and input.
        """
        grad_input = np.matmul(grad_output, self.W.T)
        grad_W = np.matmul(input.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)
        return grad_input, grad_W, grad_b


class Identity:
    def __init__(self):
        pass
    def forward(self, input):
        output = input
        return output
    def backward(self, input, grad_output):
        grad_input = grad_output
        return grad_input

class MSE_Loss:
    def __init__(self):
        pass
    def forward(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]



class MLP2:
    def __init__(self, n_feature=1, n_iter=200, lr=10, tol=None, momentum = 0.9):
        self.fc1 = Linear(n_feature, 2, bias=True)
        self.act1 = Identity()
        self.fc2 = Linear(2, n_feature, bias=True)
        self.act2 = Identity()
        self.loss_fn = MSE_Loss()
        self.n_iter = n_iter  # Maximum iteration steps
        self.lr = lr  # Learning rate
        self.tol = tol  # Tolerance for stopping iteration
        self.loss = []  # Record of loss values
        self.best_loss  = np.inf
        self.patience = 20
        self.momentum = momentum
        self.v_W1 = np.zeros_like(self.fc1.W)
        self.v_b1 = np.zeros_like(self.fc1.b)
        self.v_W2 = np.zeros_like(self.fc2.W)
        self.v_b2 = np.zeros_like(self.fc2.b)
    

    def forward(self, X):
        """
        Forward propagation through the network.
        """
        output1 = self.fc1.forward(X)
        output1a = self.act1.forward(output1)
        output2 = self.fc2.forward(output1)
        return output1,output1a,output2

    def batch_update(self, X, y):
        """
        Batch gradient descent to update weights.
        """
        epoch_not_improve = 0

        for iter in range(self.n_iter):
            output1, output1a, output2 = self.forward(X)
            y_pred = output2
            # print(f"shape is {y_pred.shape}")
            loss = self.loss_fn.forward(y_pred, y)
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

            # Backward propagation
            # grad_output3a = self.loss_fn.backward(y_pred, y)

            grad_output2a = self.loss_fn.backward(y_pred, y)
            grad_output2 = self.act2.backward(output2, grad_output2a)
            grad_output1a, grad_W2, grad_b2 = self.fc2.backward(output1a, grad_output2)
            grad_output1 = self.act1.backward(output1, grad_output1a)
            _, grad_W1, grad_b1 = self.fc1.backward(X, grad_output1)

            self.v_W1 = self.momentum * self.v_W1 + (1 - self.momentum) * grad_W1
            self.v_b1 = self.momentum * self.v_b1 + (1 - self.momentum) * grad_b1
            self.v_W2 = self.momentum * self.v_W2 + (1 - self.momentum) * grad_W2
            self.v_b2 = self.momentum * self.v_b2 + (1 - self.momentum) * grad_b2

            # Update weights
            self.fc1.W -= self.lr * self.v_W1
            self.fc1.b -= self.lr * self.v_b1
            self.fc2.W -= self.lr * self.v_W2
            self.fc2.b -= self.lr * self.v_b2

    def train(self, X_train, y_train):
        """
        Train the model using batch gradient descent.
        """
        self.batch_update(X_train, y_train)

    def predict(self, X):
        """
        Make predictions for the given input.
        """
        _, _, output2 = self.forward(X)
        return np.argmax(output2, axis=1)


    def plot_loss(self):
        """
        Plot the loss curve over training iterations.
        """
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid()
        plt.show()

    def evaluate(self, y_test, y_pred):
        """
        Evaluate the model's performance.
        """
        t = np.sum(np.equal(y_test, y_pred))
        A = (t) / len(y_test)
        return A          

if __name__ == '__main__':
    df_wine = pd.read_csv('HW9\wine.data', header=None)
    X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    _,n_feature = X_train.shape
    model = MLP2(n_feature=n_feature,n_iter=500,lr=1e-4,tol=None)
    model.train(X_train,X_train)
    y_pred = model.predict(X_test)
    print(f"Predicted labels are{y_pred}")
    print(f"Real labels are{X_test}")
    np.savetxt('HW8\pred.txt', y_pred, fmt='%f', delimiter=',')
    np.savetxt('HW8\_test.txt', X_test, fmt='%f', delimiter=',')
    plt.figure()
    model.plot_loss()
    A = model.evaluate(y_test,y_pred)
    print(f"A is {A}")

    plt.figure(figsize=(10, 5))

    colors = ['r','b']
    markers = ['s','x']

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
        plt.scatter(y_pred[y_train==l, 0], y_pred[y_train==l, 1], c=c, label=l, marker=m)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()
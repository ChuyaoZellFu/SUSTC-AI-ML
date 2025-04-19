import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class Linear:
    def __init__(self, input_dim, output_dim, bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim)*0.01

        if bias:
            self.b = np.random.randn(output_dim)*0.01
        else:
            self.b = np.zeros(output_dim)

    def forward(self, input):
        """
        Forward propagation: Z = XW + b
        """
        output = np.matmul(input, self.W) + self.b
        return output

    def backward(self, input, grad_output):
        """
        Backward propagation: Compute gradients for W, b, and input.
        """
        grad_input = np.matmul(grad_output, self.W.T)
        grad_W = np.matmul(input.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)
        return grad_input, grad_W, grad_b


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, in_put):
        """
        Sigmoid activation function: 1 / (1 + exp(-x))
        """
        in_put = np.clip(in_put, -500, 500)
        output = 1 / (1 + np.exp(-in_put))
        return output

    def backward(self, input, grad_output):
        """
        Backward propagation for sigmoid: derivative of sigmoid * grad_output
        """
        input = np.clip(input, -500, 500) 
        sigmoid = 1 / (1 + np.exp(-input))
        grad_input = grad_output * sigmoid * (1 - sigmoid)
        return grad_input
    
class ReLU:
    def __init__(self):
        pass

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, input, grad_output):
        grad_input = grad_output * (input > 0)
        return grad_input


class BCE_loss:
    def __init__(self):
        pass

    def forward(self, output, target):
        """
        Binary cross-entropy loss function.
        """
        epsilon = 1e-5
        loss = -np.mean(
            target * np.log(output + epsilon) + (1 - target) * np.log(1 - output + epsilon)
        )
        return loss

    def backward(self, output, target):
        """
        Gradient of binary cross-entropy loss w.r.t output.
        """
        epsilon = 1e-5
        grad_output = (output - target) / (epsilon + output * (1 - output))
        return grad_output


class MLP3:
    def __init__(self, n_feature=1, n_iter=200, lr=10, tol=None, momentum = 0.9):
        self.fc1 = Linear(n_feature, 32, bias=True)
        self.act1 = ReLU()
        self.fc2 = Linear(32, 32, bias=True)
        self.act2 = ReLU()
        self.fc3 = Linear(32, 1, bias=True)
        self.act3 = Sigmoid()
        self.loss_fn = BCE_loss()
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
        self.v_W3 = np.zeros_like(self.fc3.W)
        self.v_b3 = np.zeros_like(self.fc3.b)
    

    def forward(self, X):
        """
        Forward propagation through the network.
        """
        output1 = self.fc1.forward(X)
        output1a = self.act1.forward(output1)
        output2 = self.fc2.forward(output1a)
        output2a = self.act2.forward(output2)
        output3 = self.fc3.forward(output2a)
        output3a = self.act3.forward(output3)
        return output1, output1a, output2, output2a,output3,output3a

    def batch_update(self, X, y):
        """
        Batch gradient descent to update weights.
        """
        epoch_not_improve = 0

        for iter in range(self.n_iter):
            output1, output1a, output2, output2a,output3,output3a = self.forward(X)
            y_pred = output3a
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
            grad_output3a = self.loss_fn.backward(y_pred, y)
            grad_output3 = self.act3.backward(output3, grad_output3a)
            grad_output2a, grad_W3, grad_b3 = self.fc3.backward(output2a, grad_output3)
            grad_output2 = self.act2.backward(output2, grad_output2a)
            grad_output1a, grad_W2, grad_b2 = self.fc2.backward(output1a, grad_output2)
            grad_output1 = self.act1.backward(output1, grad_output1a)
            _, grad_W1, grad_b1 = self.fc1.backward(X, grad_output1)

            self.v_W1 = self.momentum * self.v_W1 + (1 - self.momentum) * grad_W1
            self.v_b1 = self.momentum * self.v_b1 + (1 - self.momentum) * grad_b1
            self.v_W2 = self.momentum * self.v_W2 + (1 - self.momentum) * grad_W2
            self.v_b2 = self.momentum * self.v_b2 + (1 - self.momentum) * grad_b2
            self.v_W3 = self.momentum * self.v_W3 + (1 - self.momentum) * grad_W3
            self.v_b3 = self.momentum * self.v_b3 + (1 - self.momentum) * grad_b3

            # Update weights
            self.fc1.W -= self.lr * self.v_W1
            self.fc1.b -= self.lr * self.v_b1
            self.fc2.W -= self.lr * self.v_W2
            self.fc2.b -= self.lr * self.v_b2
            self.fc3.W -= self.lr * self.v_W3
            self.fc3.b -= self.lr * self.v_b3

    def train(self, X_train, y_train):
        """
        Train the model using batch gradient descent.
        """
        self.batch_update(X_train, y_train)

    def predict(self, X):
        """
        Make predictions for the given input.
        """
        _, _,_, _, _, output3a = self.forward(X)
        return np.where(output3a >= 0.5, 1, 0)

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
        tp = np.sum((y_test == 1) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))

        A = (tp + tn) / len(y_test)
        P = tp / (tp + fp) if (tp + fp) > 0 else 0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0
        F_1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

        return A, R, P, F_1

            

if __name__ == '__main__':
    data = pd.read_csv('Project_Mid/ai4i2020.csv')
    print(data.isnull().sum())
    data.drop(['UDI', 'Product ID', 'Type','TWF','HDF','PWF','OSF','RNF'], axis=1, inplace=True)
    data['Tool wear rate'] = data['Tool wear [min]'] / data['Rotational speed [rpm]']
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]','Tool wear rate']
    X = data[features].values
    X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
    y = data['Machine failure'].values
    y = y[:,np.newaxis]
    # np.savetxt('project\yy.txt',y,delimiter=',',fmt='%d')
    # print(f"shape of y is {y.shape}")

    random.seed(42)
    As=[]
    Rs=[]
    Ps=[]
    F_1s=[]
    k = 5
    indices = list(range(len(X)))
    random.shuffle(indices)
    fold_size = len(X) // k
    fold_results = []
    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size if fold < k - 1 else len(X)
        test_indices = indices[start:end]
        train_indices = [i for i in indices if i not in test_indices]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        _,n_feature = X.shape
        mlp = MLP3(n_feature=n_feature,n_iter=1000,lr=5e-4,tol=None)
        mlp.train(X_train,y_train)
        mlp.plot_loss()
        y_pred = mlp.predict(X_test)
        # print(f"y_pred is {y_pred}")
        print(f"y_test is  {y_test.shape}")
        A,R,P,F_1 = mlp.evaluate(y_test, y_pred)
        As.append(A)
        Rs.append(R)
        Ps.append(P)
        F_1s.append(F_1)
        print(f"A is {A}, R is {R}, P is {P}, F1 is {F_1}")
        # print(f"Fold {fold + 1}:")
        # print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    print("A:", As)
    print("Mean A:", np.mean(As))
    print("R:", Rs)
    print("Mean R:", np.mean(Rs))
    print("P:", Ps)
    print("Mean P:", np.mean(Ps))
    print("F_1:", F_1s)
    print("Mean F_1:", np.mean(F_1s))
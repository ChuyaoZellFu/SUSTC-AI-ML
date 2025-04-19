import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import math

class Logistic:

    def __init__(self, n_feature, n_iter, lr, tol,momentum = 0.9):
        """
        Initialize the Logistic Regression model with given parameters.
        
        Parameters:
        n_feature (int): Number of features in the dataset.
        n_iter (int): Number of iterations for training.
        lr (float): Learning rate for gradient descent.
        tol (float): Tolerance for early stopping.
        """
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.W = np.random.rand(n_feature + 1) * 0.01  # Initialize weights
        self.loss = []
        self.best_loss = np.inf
        self.patience = 10  # Early stopping patience
        self.momentum = momentum
        self.velocity = np.zeros(n_feature + 1)

    def _loss(self, y, y_pred):
        """
        Compute the loss using binary cross-entropy.
        
        Parameters:
        y (float): True label.
        y_pred (float): Predicted probability.
        
        Returns:
        float: Computed loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def _gradient(self, x_bar, y, y_pred):
        """
        Compute the gradient for the weights update.
        
        Parameters:
        x_bar (numpy array): Feature vector with bias term.
        y (float): True label.
        y_pred (float): Predicted probability.
        
        Returns:
        numpy array: Computed gradient.
        """
        return (y_pred - y) * x_bar

    def _preprocess_data(self, X):
        """
        Preprocess the data by adding a bias term.
        
        Parameters:
        X (numpy array): Feature matrix.
        
        Returns:
        numpy array: Preprocessed feature matrix with bias term.
        """
        m, n = X.shape
        X_ = np.hstack((np.ones((m, 1)), X))  # Add bias term
        return X_

    def sgd_update(self, X, y):
        """
        Perform Stochastic Gradient Descent (SGD) to update weights.
        
        Parameters:
        X (numpy array): Feature matrix with bias term.
        y (numpy array): True labels.
        """
        epoch_no_improve = 0
        for iter in range(self.n_iter):
            i = np.random.randint(0, X.shape[0])  # Random sample for SGD
            y_pred = self._predict(X[i])
            loss = self._loss(y, self._predict(X))
            self.loss.append(loss)
            if self.tol is not None:
                if loss < self.best_loss - self.tol:
                    self.best_loss = loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < self.tol:
                    epoch_no_improve += 1

                if epoch_no_improve >= self.patience:
                    print(f"Early stopping triggered at iteration {iter}")
                    break

            grad = self._gradient(X[i], y[i], y_pred)
            self.velocity = self.momentum * self.velocity + self.lr * grad
            self.W -= self.velocity  # Update weights
            # self.W -= self.lr*grad

    def _sigmoid(self, z):
        """
        Compute the sigmoid function.
        
        Parameters:
        z (float): Input value.
        
        Returns:
        float: Output of sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-z))

    def _predict(self, xbar):
        """
        Predict the probability of the positive class.
        
        Parameters:
        xbar (numpy array): Feature vector with bias term.
        
        Returns:
        float: Predicted probability.
        """
        z = xbar @ self.W
        return self._sigmoid(z)

    def train(self, X_train, y_train):
        """
        Train the logistic regression model.
        
        Parameters:
        X_train (numpy array): Training feature matrix.
        y_train (numpy array): Training labels.
        """
        X_train_bar = self._preprocess_data(X_train)
        self.sgd_update(X_train_bar, y_train)

    def plot_loss(self):
        """
        Plot the loss over iterations.
        """
        plt.plot(self.loss, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()

    def evaluate(self, y_test, y_pred):
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



if __name__ == '__main__':
    
    data = pd.read_csv('project/ai4i2020.csv')
    print(data.isnull().sum())
    data.drop(['UDI', 'Product ID','TWF','HDF','PWF','OSF','RNF'], axis=1, inplace=True)
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    data.loc[data['Type']=='M','Type'] = 1
    data.loc[data['Type']=='H','Type'] = 2
    data.loc[data['Type']=='L','Type'] = 3
    data_sub = data[data['Machine failure']==1]
    for i in range(1,10):
        data = pd.concat([data,data_sub],ignore_index=True)
    X = data[features].values
    # X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
    y = data['Machine failure'].values

    np.random.seed(42)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    train_size = int(0.7 * len(y))
    test_size = len(y) - train_size
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    _,n_feature = X_train.shape
    model = Logistic(n_feature=n_feature,n_iter=30000,lr=2e-6,tol=None)
    model.train(X_train,y_train)
    print(f"Learned weights are {model.W}")
    y_pred = np.array([])
    X_test = model._preprocess_data(X_test)
    for i,x in enumerate(X_test):
        y_pred = np.append(y_pred,model._predict(x))
    print(f"Predicted labels are{y_pred}")
    y_pred = np.where(y_pred>=0.5,1,0)
    print(f"Predicted labels are{y_pred}")
    print(f"Real labels are{y_test}")
    np.savetxt('pred.txt', y_pred, fmt='%f', delimiter=',')
    np.savetxt('test.txt', y_test, fmt='%f', delimiter=',')
    plt.figure()
    model.plot_loss()
    A,R,P,F_1 = model.evaluate(y_test,y_pred)
    print(f"A is {A}, R is {R}, P is {P}, F_1 is {F_1}")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class Perceptron:

    def __init__(self,n_feature,n_iter,lr,tol,momentum = 0.9):
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.W = np.random.random(n_feature+1)*0.005
        self.loss = []
        self.best_loss = np.inf
        self.patience = 200
        self.momentum = momentum
        self.velocity = np.zeros(n_feature + 1)

    def _loss(self,y,y_pred):
        losses = np.maximum(0, -y * y_pred)  
        return np.mean(losses)  
    
    def _gradient(self,x_bar,y,y_pred):
        return -y*x_bar if y_pred * y <= 0 else 0
    
    def _preprocess_data(self,X):
        m,n = X.shape
        X_ = np.empty([m,n+1])
        X_[:,0] = 1
        X_[:,1:] = X
        return X_

    def sgd_update(self,X,y):
        epoch_no_improve = 0

        for iter in range(self.n_iter):
            i = random.randint(0,X.shape[0]-1)
            y_pred = self._predict(X[i,:])
            loss = self._loss(y,self._predict(X))
            self.loss.append(loss)

            if self.tol is not None:
                if loss < self.best_loss - self.tol:
                    self.best_loss = loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < self.tol:
                    epoch_no_improve += 1
                        
                    if epoch_no_improve >= self.patience:
                        print(f"Early stopping triggered due to no improvement in loss.")
                        break
                else:
                    epoch_no_improve = 0

            grad = self._gradient(X[i,:],y[i],y_pred)
            # self.W = self.W - self.lr*grad
            self.velocity = self.momentum * self.velocity + self.lr * grad
            self.W -= self.velocity
        
    
    def _predict(self,X):
        return X @ self.W
    
    def predict(self,X):
        X = self._preprocess_data(X)
        return X @ self.W

    def train(self,X_train,t_train):
        X_train_bar = self._preprocess_data(X_train)
        print(f"X_train_bar is {X_train_bar}\n")
        self.sgd_update(X_train_bar,t_train)

    def plot_loss(self):
        plt.plot(self.loss)
        plt.grid()
        plt.show()

    def evaluate(self,y_test,y_pred):
        tp = np.sum((y_test == 1) & (y_pred == 1))
        tn = np.sum((y_test == -1) & (y_pred == -1))
        fp = np.sum((y_test == -1) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == -1))

        A = (tp + tn) / len(y_test)
        P = tp / (tp + fp) if (tp + fp) > 0 else 0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0
        F_1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

        return A, R, P, F_1

if __name__ == '__main__':
    data = pd.read_csv('project/ai4i2020.csv')
    print(data.isnull().sum())
    data.drop(['UDI', 'Product ID','TWF','HDF','PWF','OSF','RNF'], axis=1, inplace=True)
    data['Tool wear rate'] = data['Tool wear [min]'] / data['Rotational speed [rpm]']
    data.loc[data['Type']=='M','Type'] = 1
    data.loc[data['Type']=='H','Type'] = 2
    data.loc[data['Type']=='L','Type'] = 3
    data_sub = data[data['Machine failure']==1]
    for i in range(1,20):
        data = pd.concat([data,data_sub],ignore_index=True)
        
    features = ['Type','Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]','Tool wear rate']
    X = data[features].values
    y = data['Machine failure'].values
    X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
    y[y == 0] = -1
    
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
    # X_sub = X_train[y_train==1]
    # y_sub = y_train[y_train==1]
    # for i in range(1,10):
    #     #X_add = X_sub + np.random.normal(0,0.05)
    #     X_train = np.vstack((X_train,X_sub))
    #     y_train = np.append(y_train,y_sub)
    # num_zeros = np.sum(y_train == -1)

    # # 随机选择与y==1相同数量的y==0的数据点
    # X_zeros = X_train[y_train == -1]
    # print(X_zeros)
    # indices_zeros = np.random.choice(len(X_zeros), size=len(y_sub*7), replace=False)
    # X_train = np.concatenate((X_zeros[indices_zeros], X_sub))

    # # 更新y_train，只包含选中的y==0和y==1的数据点
    # y_train = np.concatenate((y_train[indices_zeros], y_sub))

    np.savetxt('project/X_train.csv', X_train, delimiter=',', fmt='%f')
    np.savetxt('project/X_test.csv', X_test, delimiter=',', fmt='%f')
    np.savetxt('project/y_train.csv', y_train, delimiter=',', fmt='%f')
    np.savetxt('project/y_test.csv', y_test, delimiter=',', fmt='%f')

    _,n_feature = X_train.shape
    model = Perceptron(n_feature=n_feature,n_iter=35000,lr=4e-5,tol=None)
    model.train(X_train,y_train)
    print(f"Learned weights are {model.W}")
    y_pred = np.sign(model.predict(X_test))
    print(f"Predicted labels are{y_pred}")
    print(f"Real labels are{y_test}")
    plt.figure()
    model.plot_loss()
    A,R,P,F_1 = model.evaluate(y_test,y_pred)
    print(f"A is {A}, R is {R}, P is {P}, F_1 is {F_1}")




    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import math

class Logistic:

    def __init__(self,n_feature,n_iter,lr,tol):
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.W = np.random.random(n_feature+1)*0.0005
        print(f"W is {self.W}\n")
        self.loss = []
        self.best_loss = np.inf
        self.patience = 20

    def _loss(self,y,y_pred):
        epsilon = 1e-5
        return -y*math.log(y_pred+epsilon)-(1-y)*(1-math.log(y_pred+epsilon))
    
    def _gradient(self,x_bar,y,y_pred):
        return -(y-y_pred)*x_bar
    
    def _preprocess_data(self,X):
        m,n = X.shape
        X_ = np.empty([m,n+1])
        X_[:,0] = 1
        X_[:,1:] = X
        return X_

    def sgd_update(self,X,y):
        break_out = False
        epoch_no_improve = 0

        for iter in range(self.n_iter):
            i = np.random.randint(0,X.shape[0])
            y_pred = self._predict(X[i,])
            #print(f"y_pred is {y_pred},y[i] is {y[i]}\n")
            loss = self._loss(y[i],y_pred)
            self.loss.append(loss)
            if self.tol is not None:
                if loss < self.best_loss - self.tol:
                    self.best_loss = loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < self.tol:
                    epoch_no_improve += 1
                        
                    if epoch_no_improve >= self.patience:
                        print(f"Early stopping triggered due to no improvement in loss.")
                        break_out = True
                        break
                else:
                    epoch_no_improve = 0

                #print(f"x is {x}\n")
                grad = self._gradient(X[i,],y[i],y_pred)
                self.W = self.W - self.lr*grad
            if break_out:
                break_out = False
                break

    def _sigmoid(self,z):
        return 1./(1.+np.exp(-z))
    
    def _predict(self,xbar):
        z = xbar@self.W
        return self._sigmoid(z)

    def train(self,X_train,t_train):
        X_train_bar = self._preprocess_data(X_train)
        print(f"X_train_bar is {X_train_bar}\n")
        self.sgd_update(X_train_bar,t_train)
        #print(f"W is {self.W}\n")

    def plot_loss(self):
        plt.plot(self.loss)
        plt.grid()
        plt.show()

    def evaluate(self,y_test,y_pred):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0,len(y_test)):
            if(y_test[i] == y_pred[i]):
                if(y_pred[i] == 1):
                    tp = tp + 1
                if(y_pred[i] == 2):
                    tn = tn + 1
            else:
                if(y_pred[i] == 1):
                    fp = fp + 1
                if(y_pred[i] == 2):
                    fn = fn + 1
        A = (tp + tn)/(tp + fp + fn + tn)
        R = tp / (tp + fn)
        P = tp / (tp + fp)
        F_1 = 2*P*R/(P + R)
        return A,R,P,F_1

if __name__ == '__main__':
    
    data = pd.read_csv('HW4\wine.data', header=None)
    class_to_remove = 3
    filtered_data = data[data.iloc[:, 0] != class_to_remove]
    X = filtered_data.iloc[:, 1:]  
    y = filtered_data.iloc[:, 0]
    y[y == 2] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train.to_csv('HW4\X_train.data', index=False, header=False)
    y_train.to_csv('HW4\y_train.data', index=False, header=False)
    X_test.to_csv('HW4\X_test.data', index=False, header=False)
    y_test.to_csv('HW4\y_test.data', index=False, header=False)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    #print(f"X is {X_train}\n")
    
    _,n_feature = X_train.shape
    model = Logistic(n_feature=n_feature,n_iter=10000,lr=0.00001,tol=1.0e-6)
    model.train(X_train,y_train)
    print(f"Learned weights are {model.W}")

    y_pred = np.array([])
    X_test = model._preprocess_data(X_test)
    #print(f"X_test is {X_test}")
    for i,x in enumerate(X_test):
        #print(f"x is {x}")
        y_pred = np.append(y_pred,model._predict(x))
   
    y_test[y_test == 0] = 2
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5 ] = 2
    print(f"Predicted labels are{y_pred}")

    plt.figure()
    model.plot_loss()
    A,R,P,F_1 = model.evaluate(y_test,y_pred)
    print(f"A is {A}, R is {R}, P is {P}, F_1 is {F_1}")

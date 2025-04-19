import numpy as np
import matplotlib.pyplot as plt
import math

class LinearRegression:
    def __init__(self,n_feature=1,n_iter=200,lr=1e-3,tol=None):
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.W = np.random.random(n_feature+1)*0.05
        #print(self.W)
        self.loss = []

    def min_max(self,x):
        return (x-np.min(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0))
    
    def mean(self,x):
        mu = np.mean(x,axis=0)
        sigma = np.std(x,axis=0)
        return (x-mu)/sigma
    
    def _preprossdata(self,x):
        m,n = x.shape
        x_ = np.empty([m,n+1])
        x_[:,0] = 1
        x_[:,1:] = x
        #print(f'x_ is {x_}')
        return x_

    def _lossMSE(self,y,y_pred):
        return np.sum((y-y_pred)**2)/len(y)
    
    def _gradMBGD(self,x,y,y_pred):
        return (y_pred-y)@x/y.size

    def gradupdate(self,x,y):
        if self.tol is not None:
            loss_old = np.inf
    
        #m=int(x.size/20) 总数是100个，定义一组是5个（这是一个超参数，我此处的指定有些随机性）
        
        for iter in range(0,self.n_iter):
            index = np.arange(100).reshape(-1)
            np.random.shuffle(index)#打乱索引序列，实现每次取出的组中的样本都是随机的
            for n in range(0,20):
                y_pred = self._predict(x[index[5*n:5*(n+1)],:])#根据随机取出的样本进行预测以及梯度下降
                loss = self._lossMSE(y[index[5*n:5*(n+1)]],y_pred)
                self.loss.append(loss)

                if self.tol is not None:
                    if np.abs(loss_old - loss) < self.tol:
                        return iter*20
                loss_old = loss

                grad = self._gradMBGD(
                    x[index[5*n:5*(n+1)],:],y[index[5*n:5*(n+1)]],y_pred)
                self.W = self.W - self.lr*grad
        return 20*self.n_iter

    def _predict(self,x): 
        return x@self.W
    
    def predict(self,x):
        x = self._preprossdata(x)
        return x@self.W
    
    def train(self,x_train,y_train):
        #x_train = self.mean(x_train)
        x_train = self.min_max(x_train)
        #y_train = self.mean(y_train)
        #y_train = self.min_max(y_train)
        x_train = self._preprossdata(x_train)
        n_iter = self.gradupdate(x_train,y_train)
        return n_iter

    def plot_loss(self):
        plt.plot(self.loss)
        plt.grid()
        plt.show()

if __name__ == '__main__':
    x_train = np.arange(100).reshape(100,1) 
    a, b = 1, 10
    y_train = a * x_train + b  + np.random.normal(0, 5, size=x_train.shape)
    y_train = y_train.reshape(-1)
    _,n_feature = x_train.shape
    lr = 0.0008
    #print(n_feature)

    linearclassfier_1 = LinearRegression(n_feature=n_feature,n_iter=4000,lr=lr,tol=0.00001)
    n_iter = linearclassfier_1.train(x_train,y_train)
    linearclassfier_1.plot_loss()
    print(f'learning rate is {lr}')
    print(f'number of iter is {n_iter}')
    print(f'loss is {linearclassfier_1.loss[-1]}')
    print(f'learned weights are {linearclassfier_1.W}')




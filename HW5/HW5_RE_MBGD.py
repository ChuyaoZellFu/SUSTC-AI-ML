import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class LinearLayer:
    
    def __init__(self,W,activation,in_size,out_size):
        self.activation = activation
        self.W = W
        self.in_size = in_size
        self.out_size = out_size

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))    

    def forwardpass(self,x):
        self.setinput(x)
        result = x @ self.W
        if(self.activation == 'sigmoid'):
            return self.sigmoid(result)
        if(self.activation == 'tanh'):
            return np.tanh(result)

    def backwardpass(self,delta_o,output):
        if(self.activation == 'sigmoid'):
            delta_z = delta_o * output * (1-output)
        if(self.activation == 'tanh'):
            delta_z = delta_o * (1 - output**2)

        delta_h =  delta_z @ self.W.T
        return delta_z,delta_h
    def setinput(self,in_put):
        self.in_put = in_put

    def getinput(self):
        return self.in_put
    
    def setW(self,W):
        self.W = W

    def getW(self):
        return self.W



class MLP:
    def __init__(self):
        self.layers = []
        self.loss = []
        self.best_loss = np.inf
        
    def _preprocess(self,x):
        m,n = x.shape
        x_ = np.empty([m,n+1])
        x_[:,0] = 1
        x_[:,1:] = x
        return x_
    
    def min_max(self,x):
        return (x-np.min(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0))
    
    def addlayer(self,activation,in_size,out_size,add_bias):
        if(add_bias):
            W = np.random.randn(in_size+1,out_size+1)/np.sqrt(in_size+1)#Xavier Initialization
            newlayer = LinearLayer(W,activation,in_size+1,out_size+1)
        else:
            W = np.random.randn(in_size+1,out_size)/np.sqrt(in_size+1)
            newlayer = LinearLayer(W,activation,in_size+1,out_size)
        self.layers.append(newlayer)

    def forwardpass(self,x):
        in_put = x
        for layer in self.layers:
            in_put = layer.forwardpass(in_put)
        return in_put

    def getloss(self,output,y_train):
        return np.sum((output-y_train)**2)/y_train.shape[0]#Use MSE loss for regression pronblem

    def backwardpass(self,output,label,lr):
        delta_o = 2 * (label - output) * (-output)
        this_output = output
        for layer in reversed(self.layers):
            delta_z,delta_h = layer.backwardpass(delta_o,this_output)
            grad_w = layer.getinput().T @ delta_z
            layer.setW(layer.getW()-lr * grad_w)
            delta_o = delta_h
            this_output = layer.getinput()

    def fit(self,x_train,y_train,lr,iter,tol,patience):
        epoch_no_improve = 0
        x_train = self.min_max(x_train)
        y_train = self.min_max(y_train)
        x_train = self._preprocess(x_train)
        for iter in range(iter):
            pick = np.random.randint(0,x_train.shape[0])
            in_put = x_train[pick][np.newaxis,:]
            label = y_train[pick][np.newaxis,:]
            output = self.forwardpass(in_put)
            loss = self.getloss(output,label)
            self.loss.append(loss)
            if tol is not None:
                if loss < self.best_loss - tol:
                    self.best_loss = loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < tol:
                    epoch_no_improve += 1   
                    if epoch_no_improve >= patience:
                        print(f"Early stopping triggered due to no improvement in loss.")
                        break
                else:
                    epoch_no_improve = 0

            self.backwardpass(output,label,lr)

    def predict(self,x_test):
        x_test = self.min_max(x_test)
        x_test = self._preprocess(x_test)
        return self.forwardpass(x_test)
    
    def score(self,x,y):
        return self.getloss(self.predict(x),self.min_max(y))
    
    def plot_loss(self):
        plt.plot(self.loss)
        plt.grid()
        plt.show()

        
if __name__ == "__main__":
    x = np.random.uniform(low=-10.0, high=10.0, size=100)
    x = x[:,np.newaxis]
    y = 0.01*x**3
    y = y + np.random.normal(0, 0.1, y.shape)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores=[]

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        _,n_feature = x.shape
        mlp = MLP()
        mlp.addlayer('tanh',n_feature,5*n_feature,add_bias=True)
        mlp.addlayer('tanh',5*n_feature,5*n_feature,add_bias=True)
        mlp.addlayer('tanh',5*n_feature,n_feature,add_bias=False)
    
        mlp.fit(x_train,y_train,lr=3e-4,iter=50000,tol=1e-5,patience=10)
        score = mlp.score(x_test, y_test)
        scores.append(score)
    
    print("Scores:", scores)
    print("Mean score:", np.mean(scores))
    mlp.plot_loss()
    







import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import math

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
    
    def addlayer(self,activation,in_size,out_size,add_bias):
        if(add_bias):
            W = np.random.randn(in_size+1,out_size+1)/np.sqrt(in_size+1)
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
        epsilon = 1e-5
        return -y_train*math.log(output+epsilon)-(1-y_train)*(1-math.log(output+epsilon))

    def backwardpass(self,output,label,lr):
        delta_o = (output - label) / (output)*(1-output)
        this_output = output
        for layer in reversed(self.layers):
            delta_z,delta_h = layer.backwardpass(delta_o,this_output)
            grad_w = layer.getinput().T @ delta_z
            #print(f"grad_w is {grad_w}")
            layer.setW(layer.getW()-lr * grad_w)
            delta_o = delta_h
            this_output = layer.getinput()

    def fit(self,x_train,y_train,lr,iter,tol,patience):
        epoch_no_improve = 0
        x_train = self._preprocess(x_train)
        batch_size = 10
        batch_num = x_train.shape[0]/batch_size

        for iter in range(iter):
            i = np.random.randint(0,batch_num)
            output = np.array([])
            loss = 0

            if(batch_size*(i+1)>x_train.shape[0]):
                for j in range(x_train.shape[0]-batch_size*i):
                    output = np.append(output,self.forwardpass(x_train[batch_size*i+j,]))
                    loss += self.getloss(y_train[batch_size*i+j],output[j])
                
                loss = loss/(x_train.shape[0]-batch_size*i)
                self.loss.append(loss)
                self.backwardpass(output,y_train[batch_size*i:x_train.shape[0]],lr)
            else:
                for j in range(batch_size):
                    output = np.append(output,self.forwardpass(x_train[batch_size*i+j,]))
                    loss += self.getloss(y_train[batch_size*i+j],output[j])

                loss = loss/batch_size
                self.loss.append(loss)
                self.backwardpass(output,y_train[batch_size*i:batch_size*(i+1)],lr)

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


    def predict(self,x_test):
        x_test = self._preprocess(x_test)
        return self.forwardpass(x_test)
    
    def evaluate(self,y_test,y_pred):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0,len(y_test)):
            if(y_test[i] == y_pred[i]):
                if(y_pred[i] == 1):
                    tp = tp + 1
                if(y_pred[i] == 0):
                    tn = tn + 1
            else:
                if(y_pred[i] == 1):
                    fp = fp + 1
                if(y_pred[i] == 0):
                    fn = fn + 1
        A = (tp + tn)/(tp + fp + fn + tn)
        R = tp / (tp + fn)
        P = tp / (tp + fp)
        F_1 = 2*P*R/(P + R)
        return A,R,P,F_1
    
    def plot_loss(self):
        plt.plot(self.loss)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    x, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.05, random_state=42,class_sep=0.3)
    x[:, 0] = np.sin(x[:, 0])  
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
    plt.title('Non-linearly Separable Dataset')
    plt.xlabel('Transformed Feature 1')
    plt.ylabel('Transformed Feature 2')
    plt.show()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    As=[]
    Rs=[]
    Ps=[]
    F_1s=[]

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        _,n_feature = x.shape
        mlp = MLP()
        mlp.addlayer('sigmoid',n_feature,5*n_feature,add_bias=True)
        mlp.addlayer('sigmoid',5*n_feature,5*n_feature,add_bias=True)
        mlp.addlayer('sigmoid',5*n_feature,1,add_bias=False)
    
        mlp.fit(x_train,y_train,lr=7e-2,iter=10000,tol=1e-5,patience=10)
        mlp.plot_loss()
        y_pred = mlp.predict(x_test)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        print(f"y_pred is {y_pred}")
        print(f"y_test is  {y_test}")
        A,R,P,F_1 = mlp.evaluate(y_test, y_pred)
        As.append(A)
        Rs.append(R)
        Ps.append(P)
        F_1s.append(F_1)
    
    print("A:", As)
    print("Mean A:", np.mean(As))
    print("R:", Rs)
    print("Mean R:", np.mean(Rs))
    print("P:", Ps)
    print("Mean P:", np.mean(Ps))
    print("F_1:", F_1s)
    print("Mean F_1:", np.mean(F_1s))
from scipy.spatial import KDTree
import numpy as np
import pandas as pd

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.kdtree = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.kdtree = KDTree(X)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        dist, idx = self.kdtree.query(x, k=self.k, p=2)
        if idx.ndim == 1:
            idx = [idx] 
        neighbors_labels = [self.y_train[i] for i in idx[0]]
        prediction = max(set(neighbors_labels), key=neighbors_labels.count)
        return prediction
    
    def evaluate(self, y_pred, y_test):
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

if __name__ == "__main__":
    data = pd.read_csv('wdbc.data', header=None)
    X = data.iloc[:, 2:]  
    X = X.values
    y = data.iloc[:, 1]
    y = y.values
    y[y == 'M'] = 1
    y[y == 'B'] = 0

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

    knn = KNN(k=10)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("预测结果:", y_pred)
    print("实际结果:", y_test)
    A,R,P,F_1 = knn.evaluate(y_test,y_pred)
    print(f"A is {A}, R is {R}, P is {P}, F_1 is {F_1}")
    


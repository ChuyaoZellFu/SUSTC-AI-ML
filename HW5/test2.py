import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.random.uniform(low=-10.0, high=10.0, size=100)
    y = 0.01*x**3
    y = y + np.random.normal(0, 0.1, y.shape)
    plt.plot(x,y)
    plt.grid()
    plt.show()
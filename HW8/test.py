import numpy as np
class Softmax:
    def __init__(self):
        pass

    def forward(self, input):
        exps = np.exp(input)
        output = exps / np.sum(exps, axis=-1, keepdims=True)
        return output

    def backward(self, input, grad_output):
        output = self.forward(input)
        grad_input = grad_output * (output) * (1 - output)
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
    
def one_of_k_encoding(labels, num_classes):
    """
    Convert an array of labels to a one-of-K encoded matrix.
    
    Parameters:
    - labels: 1D array_like, shape (n_samples,)
        The array of labels to be encoded.
    - num_classes: int
        The number of classes.
        
    Returns:
    - one_of_k: 2D array, shape (n_samples, num_classes)
        The one-of-K encoded matrix.
    """
    one_of_k = np.zeros((len(labels), num_classes))
    one_of_k[np.arange(len(labels)), labels] = 1
    return one_of_k    

if __name__ == '__main__':
    input = np.array([[1, 2, 3]])
    y_test = np.array([3,6,8]).T
    print(y_test.shape)
    y_test = one_of_k_encoding(y_test, 10)
    print(y_test)
    softmax = Softmax()
    act = BCE_loss()
    output = softmax.forward(input)
    loss = act.forward(output, y_test)
    grad_output = act.backward(output, y_test)
    grad_input = softmax.backward(input,grad_output)
    print(grad_input)
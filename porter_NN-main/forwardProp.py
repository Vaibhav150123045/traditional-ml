
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(x):

    #YOUR CODE STARTS HERE
    
    w1 = np.random.randn(x.shape[1], 4) * 0.01
    b1 = np.zeros(shape = (1,4))
    
    #Weighted sum of hidden layer
    z1 = np.dot(x, w1) + b1   
    #Output after activation on hidden layer
    a1 = np.tanh(z1)

    #Initialize the weights and biases for the output layer
    w2 = np.random.randn(4, 1)
    b2 = np.zeros(shape = (1,1))
    
    #Weighted sum of output layer
    z2 = np.dot(a1, w2) + b2   
    #Output after activation on the output layer
    a2 = sigmoid(z2)
    
    return np.round(a2,2)
    
    #YOUR CODE ENDS HERE
    
x = np.array(eval(input()))
print(forward_propagation(x)) 
